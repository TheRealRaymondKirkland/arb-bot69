#!/usr/bin/env python3
import asyncio
import json
import logging
import os
from typing import Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


def _get_clob_client():
    """Lazy-load py_clob_client since it requires the poly-trading-bot venv."""
    try:
        from py_clob_client.client import ClobClient
        pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        host = os.getenv("POLYMARKET_HOST", CLOB_API)
        chain_id = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))
        return ClobClient(host, key=pk, chain_id=chain_id)
    except ImportError:
        logger.error("py_clob_client not available. Install from poly-trading-bot venv.")
        return None


class PolymarketClient:
    def __init__(self):
        self.dry_run = os.getenv("DRY_RUN", "true").lower() != "false"
        self._clob = None

    def _clob_client(self):
        if self._clob is None:
            self._clob = _get_clob_client()
        return self._clob

    async def get_neg_risk_opportunities(
        self, client: httpx.AsyncClient, min_profit: float = 0.04
    ) -> List[dict]:
        """
        Scan Polymarket neg-risk events where sum(YES ask) across all outcomes < 1.
        All CLOB book fetches run concurrently (semaphore-limited) across all events.
        Requires ALL books to be present — partial data is skipped to avoid false arbs.
        """
        # Paginate through all active neg-risk events (not just first 100)
        events: list = []
        offset = 0
        while True:
            try:
                r = await client.get(
                    f"{GAMMA_API}/events",
                    params={"active": "true", "closed": "false", "limit": 200,
                            "neg_risk": "true", "offset": offset},
                    timeout=15,
                )
                r.raise_for_status()
                page = r.json()
            except Exception as e:
                logger.error(f"Failed to fetch Polymarket events (offset={offset}): {e}")
                break
            if not isinstance(page, list) or not page:
                break
            events.extend(page)
            offset += len(page)
            if len(page) < 200:
                break

        semaphore = asyncio.Semaphore(20)

        async def fetch_book(token_id: str) -> Optional[dict]:
            async with semaphore:
                try:
                    book_r = await client.get(
                        f"{CLOB_API}/book",
                        params={"token_id": token_id},
                        timeout=10,
                    )
                    if book_r.status_code == 404:
                        return None
                    book_r.raise_for_status()
                    asks = book_r.json().get("asks", [])
                    if not asks:
                        return None
                    return {"price": float(asks[0]["price"]), "size": float(asks[0]["size"])}
                except Exception:
                    return None

        async def check_event(event: dict) -> Optional[dict]:
            try:
                markets = event.get("markets", [])
                ob_markets = [
                    m for m in markets
                    if m.get("enableOrderBook") and m.get("active") and not m.get("closed")
                ]
                if len(ob_markets) < 2:
                    return None

                # Collect token IDs; skip any market missing them
                token_ids = []
                valid_markets = []
                for m in ob_markets:
                    tids = json_loads_safe(m.get("clobTokenIds", "[]"))
                    if tids:
                        token_ids.append(tids[0])
                        valid_markets.append(m)

                if len(valid_markets) < 2:
                    return None

                # Fetch all order books concurrently
                books = await asyncio.gather(*[fetch_book(tid) for tid in token_ids])

                # Require ALL books — partial data means the sum is wrong
                if any(b is None for b in books):
                    return None

                total_yes_ask = sum(b["price"] for b in books)
                profit = 1.0 - total_yes_ask
                if profit < min_profit:
                    return None

                market_details = [
                    {
                        "conditionId": m.get("conditionId"),
                        "question": m.get("question", "")[:60],
                        "yes_token": token_ids[i],
                        "best_ask": books[i]["price"],
                        "best_ask_size": books[i]["size"],
                    }
                    for i, m in enumerate(valid_markets)
                ]

                min_size = min(d["best_ask_size"] for d in market_details)
                total_volume = sum(float(m.get("volumeNum", 0)) for m in ob_markets)

                return {
                    "event_slug": event.get("slug", ""),
                    "title": event.get("title", ""),
                    "n_markets": len(market_details),
                    "sum_yes_ask": total_yes_ask,
                    "profit": profit,
                    "min_size": min_size,
                    "total_volume": total_volume,
                    "markets": market_details,
                }
            except Exception as e:
                logger.debug(f"Skip Poly event {event.get('slug', '?')}: {e}")
                return None

        results = await asyncio.gather(*[check_event(e) for e in events])
        opportunities = [r for r in results if r is not None]
        opportunities.sort(key=lambda x: x["profit"], reverse=True)
        return opportunities

    async def execute_neg_risk_arb(
        self,
        opportunity: dict,
        max_usdc: float = 50.0,
    ) -> dict:
        """Buy YES on every outcome in a neg-risk event."""
        markets = opportunity["markets"]
        sum_ask = opportunity["sum_yes_ask"]
        min_size = opportunity["min_size"]

        cost_per_set = sum_ask
        max_sets_by_budget = max_usdc / cost_per_set if cost_per_set > 0 else 0
        sets_to_buy = min(max_sets_by_budget, min_size)
        sets_to_buy = round(sets_to_buy, 2)

        if sets_to_buy <= 0:
            return {"status": "skipped", "reason": "no available size"}

        total_cost = sets_to_buy * sum_ask
        expected_profit = sets_to_buy * opportunity["profit"]

        logger.info(
            f"{'[DRY RUN] ' if self.dry_run else ''}Poly neg-risk arb: {opportunity['title'][:40]} "
            f"| {sets_to_buy:.2f} sets × ${sum_ask:.3f} = ${total_cost:.2f} cost "
            f"| expected profit ${expected_profit:.2f}"
        )

        if self.dry_run:
            return {
                "status": "dry_run",
                "event": opportunity["title"],
                "sets": sets_to_buy,
                "total_cost": total_cost,
                "expected_profit": expected_profit,
            }

        clob = self._clob_client()
        if not clob:
            return {"status": "error", "reason": "clob client unavailable"}

        orders_placed = []
        try:
            creds = clob.derive_api_key(nonce=0)
            clob.set_api_creds(creds)

            for m in markets:
                from py_clob_client.clob_types import OrderArgs, OrderType
                order_args = OrderArgs(
                    token_id=m["yes_token"],
                    price=m["best_ask"],
                    size=sets_to_buy,
                    side="BUY",
                )
                resp = clob.create_and_post_order(order_args)
                orders_placed.append({
                    "question": m["question"],
                    "token": m["yes_token"],
                    "status": "placed",
                    "response": resp,
                })
        except Exception as e:
            orders_placed.append({"status": "error", "reason": str(e)})
            logger.error(f"Poly execution failed: {e}")

        return {
            "status": "executed",
            "event": opportunity["title"],
            "sets": sets_to_buy,
            "total_cost": total_cost,
            "expected_profit": expected_profit,
            "orders": orders_placed,
        }


def json_loads_safe(s) -> list:
    if isinstance(s, list):
        return s
    try:
        return json.loads(s)
    except Exception:
        return []
