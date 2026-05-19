#!/usr/bin/env python3
import asyncio
import base64
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

KALSHI_BASE         = "https://api.elections.kalshi.com/trade-api/v2"
_MIN_INTERVAL       = 0.55   # ~1.8 req/sec — Kalshi enforces ~100 req/min in practice
_next_slot          = 0.0    # next allowed fire time (monotonic)
KALSHI_TAKER_FEE    = 0.07   # fee = 0.07 × contracts × price × (1 − price), entry only

_PRIVATE_KEY = None

def _load_key():
    global _PRIVATE_KEY
    if _PRIVATE_KEY is None:
        key_file = os.getenv("KALSHI_API_KEY_FILE", "kalshi_private_key.pem")
        with open(key_file, "rb") as f:
            _PRIVATE_KEY = serialization.load_pem_private_key(f.read(), password=None)
    return _PRIVATE_KEY


def _auth_headers(method: str, path: str) -> Dict[str, str]:
    key_id = os.getenv("KALSHI_API_KEY_ID", "")
    pk = _load_key()
    ts = str(int(time.time() * 1000))
    clean_path = path.split("?")[0]
    msg = (ts + method.upper() + clean_path).encode()
    sig = pk.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "Content-Type": "application/json",
    }


async def _get(client: httpx.AsyncClient, path: str, params: dict = None) -> dict:
    await _rate_limit()

    url = KALSHI_BASE + path
    if params:
        url = url + "?" + urlencode(params)
    for attempt in range(4):
        r = await client.get(url, headers=_auth_headers("GET", "/trade-api/v2" + path), timeout=15)
        if r.status_code == 429:
            wait_t = 2 ** attempt
            logger.warning(f"Kalshi 429 on {path} — retrying in {wait_t}s (attempt {attempt + 1})")
            await asyncio.sleep(wait_t)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return r.json()


async def _rate_limit():
    """Shared token-bucket slot reservation for all HTTP methods."""
    global _next_slot
    now  = time.monotonic()
    fire = max(now, _next_slot)
    _next_slot = fire + _MIN_INTERVAL
    wait = fire - time.monotonic()
    if wait > 0:
        await asyncio.sleep(wait)


async def _post(client: httpx.AsyncClient, path: str, body: dict) -> dict:
    await _rate_limit()
    for attempt in range(4):
        r = await client.post(
            KALSHI_BASE + path,
            headers=_auth_headers("POST", "/trade-api/v2" + path),
            json=body,
            timeout=15,
        )
        if r.status_code == 429:
            wait_t = 2 ** attempt
            logger.warning(f"Kalshi 429 on POST {path} — retrying in {wait_t}s")
            await asyncio.sleep(wait_t)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return r.json()


async def _delete(client: httpx.AsyncClient, path: str) -> dict:
    await _rate_limit()
    for attempt in range(4):
        r = await client.delete(
            KALSHI_BASE + path,
            headers=_auth_headers("DELETE", "/trade-api/v2" + path),
            timeout=15,
        )
        if r.status_code == 429:
            wait_t = 2 ** attempt
            logger.warning(f"Kalshi 429 on DELETE {path} — retrying in {wait_t}s")
            await asyncio.sleep(wait_t)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return r.json()


def _walk_books(
    books: List[List[list]],
    min_profit: float,
    max_profit: float,
    budget: float = 100.0,
) -> Tuple[int, float, float, float]:
    """
    Walk multi-leg Kalshi order books to find max profitable sets.

    books: per-leg YES ask levels from orderbook_fp.yes_dollars:
           [[price_str, size_str], ...] sorted ascending by price (best ask first).

    Returns (total_sets, avg_cost_per_set, total_cost, total_expected_profit).
    Stops when the marginal set is unprofitable after fees or budget is exhausted.
    """
    # Expand each leg into a flat per-contract price list
    per_leg: List[List[float]] = []
    for levels in books:
        prices: List[float] = []
        for lvl in levels:
            try:
                price = float(lvl[0])
                size  = int(float(lvl[1]))
                prices.extend([price] * size)
            except (IndexError, ValueError, TypeError):
                continue
        per_leg.append(prices)

    if not per_leg or not all(per_leg):
        return 0, 0.0, 0.0, 0.0

    max_depth    = min(len(leg) for leg in per_leg)
    total_sets   = 0
    total_cost   = 0.0
    total_profit = 0.0

    for k in range(max_depth):
        marginal = [leg[k] for leg in per_leg]
        cost_k   = sum(marginal)
        fee_k    = KALSHI_TAKER_FEE * sum(p * (1.0 - p) for p in marginal)
        profit_k = 1.0 - cost_k - fee_k

        if profit_k < min_profit:
            break
        if total_cost + cost_k > budget:
            break

        total_sets   += 1
        total_cost   += cost_k
        total_profit += profit_k

    avg_cost = total_cost / total_sets if total_sets > 0 else 0.0
    return total_sets, round(avg_cost, 4), round(total_cost, 4), round(total_profit, 6)


class KalshiClient:
    def __init__(self):
        self.dry_run = os.getenv("DRY_RUN", "true").lower() != "false"

    async def get_balance(self, client: httpx.AsyncClient) -> float:
        data = await _get(client, "/portfolio/balance")
        return data.get("balance", 0) / 100.0

    async def get_event_markets(self, client: httpx.AsyncClient, event_ticker: str) -> List[dict]:
        data = await _get(client, "/markets", {"event_ticker": event_ticker, "limit": 200, "status": "open"})
        return [m for m in data.get("markets", []) if m.get("result") == "" and m.get("status") == "active"]

    async def get_all_mece_events(self, client: httpx.AsyncClient) -> List[dict]:
        """Return open mutually-exclusive events via dynamic series discovery."""
        from kalshi.series_discovery import get_mece_events
        return await get_mece_events(client)

    async def _check_event(self, client: httpx.AsyncClient, event: dict, min_profit: float, max_profit: float = 0.15) -> Optional[dict]:
        """Check a single event for arb — called concurrently."""
        try:
            markets = await self.get_event_markets(client, event["event_ticker"])
            if len(markets) < 2:
                return None

            prices = [m.get("yes_ask_dollars") for m in markets]
            if any(p is None for p in prices):
                logger.debug(f"Skip {event['event_ticker']}: missing price data")
                return None
            fp = [float(p) for p in prices]
            sum_yes_ask  = sum(fp)
            gross_profit = 1.0 - sum_yes_ask
            # Entry taker fee per set: 0.07 × price × (1 − price) per leg
            fee_per_set  = sum(KALSHI_TAKER_FEE * p * (1.0 - p) for p in fp)
            profit       = gross_profit - fee_per_set  # net profit after fees

            if profit < min_profit or profit > max_profit:
                return None

            # Filter by close_time — skip markets that resolve too far out
            max_days = float(os.getenv("MAX_DAYS_TO_CLOSE", "7"))
            now_utc = datetime.now(timezone.utc)
            close_times = []
            for m in markets:
                ct = m.get("close_time")
                if ct:
                    try:
                        close_times.append(datetime.fromisoformat(ct.replace("Z", "+00:00")))
                    except Exception:
                        pass
            if not close_times:
                return None
            earliest_close = min(close_times)
            days_to_close = (earliest_close - now_utc).total_seconds() / 86400
            if days_to_close < 0 or days_to_close > max_days:
                return None

            sizes = [float(m.get("yes_ask_size_fp", 0)) for m in markets]
            min_size = min(sizes) if sizes else 0
            if min_size < 0.01:
                return None

            titles = [m.get("yes_sub_title", "").lower() for m in markets]
            if len(markets) == 2:
                has_catchall = True
            else:
                _CATCHALL_KW = (
                    "no new", "other", "none", "someone else", "stays", "no change",
                    "field ", "undeclared", "another candidate", "nobody", "no one",
                    "anyone else", "somebody else", "alternative",
                )
                has_catchall = any(
                    any(kw in t for kw in _CATCHALL_KW)
                    for t in titles
                )

            total_vol = sum(float(m.get("volume_fp", 0)) for m in markets)

            return {
                "event_ticker":  event["event_ticker"],
                "title":         event.get("title", ""),
                "n_markets":     len(markets),
                "sum_yes_ask":   sum_yes_ask,
                "gross_profit":  gross_profit,
                "fee_per_set":   fee_per_set,
                "profit":        profit,       # net after fees
                "min_size":      min_size,
                "total_volume":  total_vol,
                "has_catchall":  has_catchall,
                "days_to_close": round(days_to_close, 1),
                "markets":       markets,
            }
        except Exception as e:
            logger.debug(f"Skip {event['event_ticker']}: {e}")
            return None

    async def _deepen_opportunity(
        self,
        client: httpx.AsyncClient,
        opp: dict,
        min_profit: float,
        max_profit: float,
        budget: float = 100.0,
        book_depth: int = 25,
    ) -> dict:
        """
        Fetch live order books for every leg of an opportunity and walk them
        to find the max capital we can deploy while staying profitable.
        Adds book_max_sets / book_avg_cost / book_total_cost / book_total_profit.
        """
        markets = opp.get("markets", [])
        sem = asyncio.Semaphore(5)

        async def _fetch_leg(ticker: str) -> List[list]:
            async with sem:
                try:
                    data = await _get(client, f"/markets/{ticker}/orderbook", {"depth": book_depth})
                    ob = data.get("orderbook_fp", {})
                    # yes_dollars = YES bids (sorted ascending, worst→best).
                    # To BUY YES we lift NO bids: YES ask = 1 - NO bid.
                    # Reverse no_dollars so cheapest YES ask comes first.
                    no_bids = ob.get("no_dollars", [])
                    return [
                        [str(round(1.0 - float(lvl[0]), 4)), lvl[1]]
                        for lvl in reversed(no_bids)
                        if lvl and len(lvl) >= 2
                    ]
                except Exception as e:
                    logger.debug(f"Orderbook fetch failed {ticker}: {e}")
                    return []

        books = await asyncio.gather(*[_fetch_leg(m.get("ticker", "")) for m in markets])
        sets, avg_cost, total_cost, total_profit = _walk_books(
            list(books), min_profit, max_profit, budget=budget
        )

        result = dict(opp)
        result["book_max_sets"]     = sets
        result["book_avg_cost"]     = avg_cost
        result["book_total_cost"]   = total_cost
        result["book_total_profit"] = total_profit
        return result

    async def scan_mece_opportunities(
        self, client: httpx.AsyncClient, min_profit: float = 0.04, max_profit: float = 0.15
    ) -> List[dict]:
        """Check each MECE event for arb concurrently, rate-limited by _get's token bucket."""
        events = await self.get_all_mece_events(client)
        logger.info(f"Scanning {len(events)} MECE events for arb ({min_profit:.0%}–{max_profit:.0%})...")

        sem = asyncio.Semaphore(5)

        async def _bounded_check(event):
            async with sem:
                return await self._check_event(client, event, min_profit, max_profit)

        results = await asyncio.gather(*[_bounded_check(e) for e in events])
        opportunities = [r for r in results if r is not None]

        # Walk order books for each found opportunity to size capital deployment
        if opportunities:
            budget = float(os.getenv("MAX_POSITION_USDC", "100"))
            logger.info(f"Walking order books for {len(opportunities)} opportunities (budget=${budget:.0f})...")
            deepened = []
            for opp in opportunities:
                deepened.append(await self._deepen_opportunity(
                    client, opp, min_profit, max_profit, budget=budget
                ))
            opportunities = deepened

        opportunities.sort(key=lambda x: (x["has_catchall"], x["book_total_profit"]), reverse=True)
        logger.info(f"Found {len(opportunities)} MECE arb opportunities")
        return opportunities

    async def execute_mece_arb(
        self,
        client: httpx.AsyncClient,
        opportunity: dict,
        max_usdc: float = 50.0,
    ) -> dict:
        """
        Buy YES on every market in the opportunity up to max_usdc total.
        Returns execution summary.
        """
        markets = opportunity["markets"]
        n = len(markets)
        sum_ask = opportunity["sum_yes_ask"]

        # How many contract-sets can we buy within budget?
        cost_per_set = sum_ask
        max_sets_by_budget = max_usdc / cost_per_set if cost_per_set > 0 else 0
        max_sets_by_size = opportunity["min_size"]
        sets_to_buy = min(max_sets_by_budget, max_sets_by_size)
        sets_to_buy = int(sets_to_buy * 10000) / 10000  # floor to 4 decimals

        if sets_to_buy <= 0:
            return {"status": "skipped", "reason": "no available size"}

        total_cost = sets_to_buy * sum_ask
        expected_profit = sets_to_buy * opportunity["profit"]

        logger.info(
            f"{'[DRY RUN] ' if self.dry_run else ''}MECE arb: {opportunity['event_ticker']} "
            f"| {sets_to_buy:.2f} sets × ${sum_ask:.3f} = ${total_cost:.2f} cost "
            f"| expected profit ${expected_profit:.2f}"
        )

        orders_placed = []
        if not self.dry_run:
            placed_order_ids = []
            failed = False

            for m in markets:
                ask_price = float(m.get("yes_ask_dollars", 1))
                ticker = m["ticker"]
                try:
                    order = await _post(client, "/portfolio/orders", {
                        "ticker": ticker,
                        "client_order_id": str(uuid.uuid4()),
                        "type": "limit",
                        "action": "buy",
                        "side": "yes",
                        "count": sets_to_buy,
                        "yes_price": max(1, min(99, round(ask_price * 100))),
                    })
                    order_id = order.get("order", {}).get("order_id")
                    orders_placed.append({"ticker": ticker, "status": "placed", "order_id": order_id})
                    if order_id:
                        placed_order_ids.append(order_id)
                    logger.info(f"  Placed YES buy on {ticker} at ${ask_price:.3f} × {sets_to_buy:.2f}")
                except Exception as e:
                    orders_placed.append({"ticker": ticker, "status": "failed", "error": str(e)})
                    logger.error(f"  FAILED to place order on {ticker}: {e}")
                    failed = True
                    break  # stop placing remaining legs

            # If any leg failed, cancel everything already placed to avoid a directional position.
            if failed and placed_order_ids:
                logger.error(
                    f"  Partial execution on {opportunity['event_ticker']} — "
                    f"cancelling {len(placed_order_ids)} placed order(s)."
                )
                for oid in placed_order_ids:
                    try:
                        await _delete(client, f"/portfolio/orders/{oid}")
                        logger.info(f"  Cancelled order {oid}")
                    except Exception as ce:
                        logger.error(f"  FAILED to cancel order {oid}: {ce} — MANUAL ACTION REQUIRED")

                return {
                    "status": "partial_fail_cancelled",
                    "event": opportunity["event_ticker"],
                    "sets": sets_to_buy,
                    "total_cost": 0,
                    "expected_profit": 0,
                    "orders": orders_placed,
                }
        else:
            for m in markets:
                orders_placed.append({
                    "ticker": m["ticker"],
                    "status": "dry_run",
                    "would_buy": sets_to_buy,
                    "at_price": float(m.get("yes_ask_dollars", 0)),
                })

        return {
            "status": "dry_run" if self.dry_run else "executed",
            "event": opportunity["event_ticker"],
            "sets": sets_to_buy,
            "total_cost": total_cost,
            "expected_profit": expected_profit,
            "orders": orders_placed,
        }
