"""
Cross-platform arbitrage scanner: Kalshi vs Polymarket.

Strategy: find the same underlying event priced differently on both platforms.
When sum(YES prices for all outcomes) < $1.00 on one platform but not the other,
or when the YES price for a binary outcome differs by >MIN_SPREAD, flag it.

Two discovery modes:
  1. Manual pairs: data/cross_platform_pairs.json — curated, high-confidence matches
  2. Auto-match: keyword overlap between Kalshi MECE event titles and Polymarket
     event titles — lower confidence, flagged for review only (never auto-executed)

Execution: this scanner is REPORT ONLY. It surfaces opportunities; a human
confirms the resolution criteria match before any order is placed.
"""
import json
import logging
import re
from pathlib import Path
from typing import Optional

import httpx

from kalshi.kalshi_client import KalshiClient, _get

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"
PAIRS_PATH = Path(__file__).parent.parent / "data" / "cross_platform_pairs.json"
MIN_SPREAD = 0.03   # 3 cents minimum price gap to flag
POLY_PAGES = 5      # 5 × 200 = 1000 Polymarket events scanned per run


# ---------------------------------------------------------------------------
# Polymarket helpers
# ---------------------------------------------------------------------------

def _parse_outcome_prices(raw) -> list[float]:
    if not raw:
        return []
    if isinstance(raw, list):
        try:
            return [float(x) for x in raw]
        except Exception:
            return []
    try:
        return [float(x) for x in json.loads(raw)]
    except Exception:
        return []


async def _fetch_poly_events(client: httpx.AsyncClient) -> list[dict]:
    """Fetch all active Polymarket events with nested markets (up to POLY_PAGES pages)."""
    events = []
    offset = 0
    for _ in range(POLY_PAGES):
        try:
            r = await client.get(
                f"{GAMMA_API}/events",
                params={"active": "true", "closed": "false", "limit": 200, "offset": offset},
                timeout=20,
            )
            r.raise_for_status()
            page = r.json()
        except Exception as e:
            logger.warning(f"Polymarket event page failed: {e}")
            break
        if not isinstance(page, list) or not page:
            break
        events.extend(page)
        offset += len(page)
        if len(page) < 200:
            break
    return events


async def _fetch_clob_book(client: httpx.AsyncClient, token_id: str) -> Optional[float]:
    """Return best YES ask price from the CLOB order book for a token."""
    try:
        r = await client.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        asks = r.json().get("asks", [])
        return float(asks[0]["price"]) if asks else None
    except Exception:
        return None


def _poly_event_yes_sum(event: dict) -> tuple[float, int]:
    """
    Compute sum of YES (outcome 0) prices for all nested markets in a Polymarket event.
    Returns (sum_yes, market_count). Uses outcomePrices (snapshot) — fast, no CLOB call.
    """
    total = 0.0
    count = 0
    for m in event.get("markets", []):
        if not m.get("active") or m.get("closed"):
            continue
        prices = _parse_outcome_prices(m.get("outcomePrices"))
        if prices:
            total += prices[0]
            count += 1
    return total, count


# ---------------------------------------------------------------------------
# Title matching helpers
# ---------------------------------------------------------------------------

_STOP = {"the", "a", "an", "of", "in", "to", "for", "and", "or", "is", "will",
         "be", "by", "at", "on", "who", "what", "which", "next", "new", "us"}

def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


def _match_score(a: str, b: str) -> float:
    """Jaccard-style overlap between title token sets (0–1)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# Manual pairs file
# ---------------------------------------------------------------------------

def load_manual_pairs() -> list[dict]:
    """
    Load manually curated Kalshi ↔ Polymarket pairs from data/cross_platform_pairs.json.

    Format:
    [
      {
        "kalshi_event_ticker": "KXFED-27APR",
        "poly_event_id": "12345",
        "poly_event_slug": "fed-rate-april-2027",
        "notes": "Both resolve on FOMC April 2027 decision",
        "verified": true
      }
    ]
    """
    if not PAIRS_PATH.exists():
        return []
    try:
        return json.loads(PAIRS_PATH.read_text())
    except Exception as e:
        logger.warning(f"Failed to load pairs file: {e}")
        return []


def save_manual_pairs(pairs: list[dict]) -> None:
    PAIRS_PATH.parent.mkdir(exist_ok=True)
    PAIRS_PATH.write_text(json.dumps(pairs, indent=2))


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

class CrossPlatformScanner:
    def __init__(self, min_spread: float = MIN_SPREAD):
        self.min_spread = min_spread
        self.kalshi = KalshiClient()

    async def scan(
        self,
        client: httpx.AsyncClient,
        kalshi_opps: Optional[list] = None,
    ) -> dict:
        """
        Run the full cross-platform scan. Returns:
          {
            "poly_mece_opps":  [...],   # Polymarket-only MECE arbs
            "kalshi_opps":     [...],   # Kalshi MECE opps
            "cross_opps":      [...],   # Manual-pair price discrepancies
            "auto_candidates": [...],   # Unverified title-match candidates for review
          }

        Pass kalshi_opps from an existing Kalshi scan to avoid re-running it.
        """
        import asyncio

        poly_events_task = asyncio.create_task(_fetch_poly_events(client))

        if kalshi_opps is None:
            kalshi_opps_task = asyncio.create_task(
                self.kalshi.scan_mece_opportunities(client, min_profit=self.min_spread)
            )
            poly_events, kalshi_opps = await asyncio.gather(poly_events_task, kalshi_opps_task)
        else:
            poly_events = await poly_events_task

        poly_mece_opps = self._scan_poly_mece(poly_events)
        cross_opps = await self._scan_manual_pairs(client, poly_events)
        auto_candidates = self._auto_match(kalshi_opps, poly_events)

        return {
            "poly_mece_opps": poly_mece_opps,
            "kalshi_opps": kalshi_opps,
            "cross_opps": cross_opps,
            "auto_candidates": auto_candidates,
        }

    # ------------------------------------------------------------------
    def _scan_poly_mece(self, poly_events: list[dict]) -> list[dict]:
        """
        Find Polymarket neg-risk events where sum(YES prices) < 1 - min_spread.

        Requires neg_risk=True (Polymarket's own flag for mutually-exclusive
        outcome sets) to avoid false positives from "by when" date-window markets,
        which are correlated binaries — NOT MECE.
        """
        opps = []
        for event in poly_events:
            # Only process genuine neg-risk events where exactly one outcome wins
            if not event.get("negRisk") and not event.get("neg_risk"):
                continue

            markets = [
                m for m in event.get("markets", [])
                if m.get("active") and not m.get("closed")
                   and (m.get("enableOrderBook") or m.get("enable_order_book"))
            ]
            if len(markets) < 3:
                continue

            sum_yes, count = _poly_event_yes_sum(event)
            if count < 3:
                continue

            profit = 1.0 - sum_yes
            if profit < self.min_spread:
                continue

            titles = [m.get("question", "").lower() for m in markets]
            has_catchall = any(
                any(kw in t for kw in ["none", "other", "no new", "someone else", "neither",
                                        "no change", "stays", "no cuts", "unchanged"])
                for t in titles
            )

            opps.append({
                "platform": "Polymarket",
                "event_id": event.get("id", ""),
                "event_slug": event.get("slug", ""),
                "title": event.get("title", ""),
                "n_markets": count,
                "sum_yes": sum_yes,
                "profit": profit,
                "has_catchall": has_catchall,
                "markets": [
                    {
                        "question": m.get("question", "")[:60],
                        "yes_price": _parse_outcome_prices(m.get("outcomePrices"))[0]
                            if _parse_outcome_prices(m.get("outcomePrices")) else 0,
                        "token_id": (json.loads(m.get("clobTokenIds", "[]"))
                                     if isinstance(m.get("clobTokenIds"), str)
                                     else (m.get("clobTokenIds") or [None]))[0],
                    }
                    for m in markets
                    if _parse_outcome_prices(m.get("outcomePrices"))
                ],
            })

        opps.sort(key=lambda x: (x["has_catchall"], x["profit"]), reverse=True)
        return opps

    # ------------------------------------------------------------------
    async def _scan_manual_pairs(
        self, client: httpx.AsyncClient, poly_events: list[dict]
    ) -> list[dict]:
        """
        For each manually verified Kalshi ↔ Polymarket pair, compare YES prices.
        Returns discrepancies where the spread exceeds min_spread.
        """
        pairs = load_manual_pairs()
        if not pairs:
            return []

        poly_by_id = {str(e.get("id", "")): e for e in poly_events}
        poly_by_slug = {e.get("slug", ""): e for e in poly_events}

        results = []
        for pair in pairs:
            kalshi_ticker = pair.get("kalshi_event_ticker", "")
            poly_slug = pair.get("poly_event_slug", "")
            poly_id = str(pair.get("poly_event_id", ""))

            poly_event = poly_by_id.get(poly_id) or poly_by_slug.get(poly_slug)
            if not poly_event:
                continue

            try:
                kalshi_markets = await self.kalshi.get_event_markets(client, kalshi_ticker)
            except Exception as e:
                logger.debug(f"Cross-pair Kalshi fetch failed for {kalshi_ticker}: {e}")
                continue

            if not kalshi_markets:
                continue

            k_sum = sum(float(m.get("yes_ask_dollars", 1)) for m in kalshi_markets)
            p_sum, _ = _poly_event_yes_sum(poly_event)

            if abs(k_sum - p_sum) < self.min_spread:
                continue

            results.append({
                "kalshi_ticker": kalshi_ticker,
                "poly_slug": poly_slug,
                "notes": pair.get("notes", ""),
                "verified": pair.get("verified", False),
                "kalshi_sum_yes_ask": k_sum,
                "poly_sum_yes_ask": p_sum,
                "spread": abs(k_sum - p_sum),
                "cheaper_on": "kalshi" if k_sum < p_sum else "polymarket",
            })

        results.sort(key=lambda x: x["spread"], reverse=True)
        return results

    # ------------------------------------------------------------------
    def _auto_match(
        self, kalshi_opps: list[dict], poly_events: list[dict]
    ) -> list[dict]:
        """
        For each Kalshi MECE opportunity, find Polymarket events with high title
        token overlap. Returns candidates for human review — never auto-executed.
        Min score 0.25 to reduce noise.
        """
        candidates = []
        for k_opp in kalshi_opps:
            k_title = k_opp.get("title", "")
            best_score = 0.0
            best_match = None
            for p_event in poly_events:
                score = _match_score(k_title, p_event.get("title", ""))
                if score > best_score:
                    best_score = score
                    best_match = p_event

            if best_score >= 0.25 and best_match:
                p_sum, _ = _poly_event_yes_sum(best_match)
                candidates.append({
                    "kalshi_ticker": k_opp["event_ticker"],
                    "kalshi_title": k_title,
                    "kalshi_sum_yes_ask": k_opp["sum_yes_ask"],
                    "poly_event_id": best_match.get("id", ""),
                    "poly_title": best_match.get("title", ""),
                    "poly_sum_yes_ask": p_sum,
                    "match_score": round(best_score, 3),
                    "note": "AUTO-MATCHED — verify resolution criteria before trading",
                })

        candidates.sort(key=lambda x: x["match_score"], reverse=True)
        return candidates


def print_cross_platform_report(results: dict) -> None:
    poly_opps = results.get("poly_mece_opps", [])
    cross_opps = results.get("cross_opps", [])
    auto_cands = results.get("auto_candidates", [])

    print(f"\n  {'─'*56}")
    print(f"  CROSS-PLATFORM SCAN")
    print(f"  {'─'*56}")

    # Polymarket MECE opps
    print(f"\n  Polymarket MECE opportunities: {len(poly_opps)}")
    for i, opp in enumerate(poly_opps[:5], 1):
        safe = "✅ COMPLETE" if opp["has_catchall"] else "⚠️  INCOMPLETE"
        print(f"\n  #{i} [Polymarket] {opp['event_slug'][:50]}")
        print(f"     {opp['title'][:60]}")
        print(f"     Markets: {opp['n_markets']}  sum_YES: ${opp['sum_yes']:.4f}  "
              f"profit: ${opp['profit']:.4f} ({opp['profit']*100:.1f}%)  {safe}")
        for m in opp["markets"][:6]:
            print(f"       {m['question']:<50} YES=${m['yes_price']:.3f}")

    # Manual pair discrepancies
    if cross_opps:
        print(f"\n  Manual-pair discrepancies: {len(cross_opps)}")
        for opp in cross_opps:
            arrow = "Kalshi cheaper →buy Kalshi, sell Poly" if opp["cheaper_on"] == "kalshi" \
                    else "Poly cheaper → buy Poly, sell Kalshi"
            print(f"\n  {opp['kalshi_ticker']} ↔ {opp['poly_slug']}")
            print(f"     Kalshi sum: ${opp['kalshi_sum_yes_ask']:.4f}  "
                  f"Poly sum: ${opp['poly_sum_yes_ask']:.4f}  "
                  f"spread: ${opp['spread']:.4f}")
            print(f"     {arrow}")
            if not opp["verified"]:
                print(f"     ⚠️  Not verified — confirm resolution criteria match")

    # Auto-match candidates (review only)
    if auto_cands:
        print(f"\n  Auto-matched candidates (review only): {len(auto_cands)}")
        for c in auto_cands[:3]:
            print(f"\n  Kalshi: {c['kalshi_ticker']} — {c['kalshi_title'][:50]}")
            print(f"  Poly:   {c['poly_event_id']} — {c['poly_title'][:50]}")
            print(f"  Kalshi sum=${c['kalshi_sum_yes_ask']:.3f}  Poly sum=${c['poly_sum_yes_ask']:.3f}  "
                  f"match_score={c['match_score']}")
            print(f"  ⚠️  {c['note']}")
