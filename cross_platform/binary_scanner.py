"""
Cross-platform binary arbitrage scanner: Kalshi vs Polymarket.

For binary YES/NO markets listed on both platforms:
  Direction 1 — Buy Kalshi YES + Buy Polymarket NO:
      profitable when  YES_ask(K) + NO_ask(P) < $1.00
  Direction 2 — Buy Polymarket YES + Buy Kalshi NO:
      profitable when  YES_ask(P) + NO_ask(K) < $1.00

Workflow:
  1. Fetch all active binary markets from both platforms concurrently.
  2. Match by title using the improved scorer in matcher.py.
  3. Fast pre-filter using Polymarket snapshot prices (no extra API calls).
  4. Fetch precise Polymarket CLOB prices only for promising pairs.
  5. Report confirmed arbs and the full candidate list.
"""
import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import httpx

from kalshi.kalshi_client import _get
from cross_platform.matcher import match_score

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

MIN_MATCH_SCORE   = 0.45   # raised from 0.28 — drops year-only matches (London/2028 vs US/2028)
MIN_PROFIT        = 0.03   # 3% minimum to flag as opportunity
SNAPSHOT_BUFFER   = 0.02   # skip CLOB fetch if snapshot gap < this (saves API calls)
KALSHI_MAX_PAGES  = 5      # 5 × 200 = up to 1 000 markets
POLY_MAX_PAGES    = 10     # 10 × 200 = up to 2 000 markets
CLOB_SEMAPHORE    = 15


@dataclass
class BinaryArb:
    kalshi_ticker:  str
    kalshi_title:   str
    poly_market_id: str
    poly_question:  str
    score:          float

    direction:      str    # "buy_kalshi_yes_poly_no" | "buy_poly_yes_kalshi_no"
    kalshi_leg:     float  # price paid on Kalshi side
    poly_leg:       float  # price paid on Polymarket side
    profit_pct:     float  # profit as a fraction (0.05 = 5%)
    max_contracts:  float  # real depth-limited contract count
    max_profit_usd: float  # max_contracts × profit_pct
    clob_verified:  bool   # True = prices from live CLOB books (not snapshots)

    k_yes_ask: float
    k_no_ask:  float
    p_yes_ask: float
    p_no_ask:  float


# ─── Kalshi ──────────────────────────────────────────────────────────────────

async def fetch_kalshi_binary_markets(
    client: httpx.AsyncClient,
    poly_questions: Optional[list[str]] = None,
) -> list[dict]:
    """
    Two-step approach:
    1. Page through /events to collect non-MECE event titles (5 API calls).
    2. Pre-filter by Polymarket title match (no API calls), then per-event
       market fetch for only the events that could plausibly match (~50 calls).

    This replaces the old 864-call-per-event approach and the bulk approach
    that couldn't reach political markets buried past Kalshi's first 15k results.
    """
    # Step 1: collect non-MECE event metadata (5 pages × 200 = 1000 events)
    event_meta: dict[str, str] = {}  # event_ticker → title
    cursor: Optional[str] = None
    for page in range(KALSHI_MAX_PAGES):
        params: dict = {"status": "open", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        try:
            data = await _get(client, "/events", params)
        except Exception as e:
            logger.warning(f"Kalshi events page {page + 1} failed: {e}")
            break
        events = data.get("events", [])
        if not events:
            break
        for e in events:
            if not e.get("mutually_exclusive"):
                event_meta[e["event_ticker"]] = e.get("title", e["event_ticker"])
        cursor = data.get("cursor")
        if not cursor:
            break

    logger.info(f"Kalshi: found {len(event_meta)} non-MECE events to check")

    # Step 2: pre-filter using Polymarket questions (pure Python, no API calls)
    PRE_FILTER_THRESHOLD = 0.35
    if poly_questions:
        candidates = []
        for et, title in event_meta.items():
            best = max((match_score(title, q)[0] for q in poly_questions), default=0)
            if best >= PRE_FILTER_THRESHOLD:
                candidates.append(et)
        logger.info(f"Kalshi: {len(candidates)} events pre-matched against Polymarket titles")
    else:
        candidates = list(event_meta.keys())

    # Step 3: per-event market fetch for candidate events only
    sem = asyncio.Semaphore(5)

    async def check_event(et: str) -> Optional[dict]:
        async with sem:
            try:
                mkt_data = await _get(client, "/markets", {
                    "event_ticker": et, "limit": 10, "status": "open"
                })
            except Exception:
                return None
            mkts = [
                m for m in mkt_data.get("markets", [])
                if m.get("result") == "" and m.get("status") == "active"
            ]
            if len(mkts) != 1:
                return None
            m = mkts[0]
            try:
                yes_ask  = float(m.get("yes_ask_dollars", 1))
                yes_bid  = float(m.get("yes_bid_dollars", 0))
                yes_size = float(m.get("yes_ask_size_fp", 0) or 0)
            except (TypeError, ValueError):
                return None
            if yes_ask >= 0.99 or yes_bid <= 0.01 or yes_ask <= 0.01:
                return None
            return {
                "event_ticker": et,
                "ticker":       m.get("ticker", ""),
                "title":        event_meta.get(et, et),
                "yes_ask":      round(yes_ask, 4),
                "yes_bid":      round(yes_bid, 4),
                "no_ask":       round(1.0 - yes_bid, 4),
                "yes_size":     yes_size,
                "volume":       float(m.get("volume_fp", 0) or 0),
            }

    results = await asyncio.gather(*[check_event(et) for et in candidates])
    binary = [r for r in results if r is not None]
    logger.info(f"Kalshi: {len(binary)} active binary markets found")
    return binary


# ─── Kalshi MECE candidates ──────────────────────────────────────────────────

def build_kalshi_mece_candidates(mece_opps: list[dict]) -> list[dict]:
    """
    Convert pre-fetched MECE scan results into individual binary market records
    so the title matcher can compare each candidate against Polymarket markets.

    Accepts the output of KalshiClient.scan_mece_opportunities() directly —
    no second API call needed.
    """
    candidates: list[dict] = []
    for ev in mece_opps:
        parent_title = ev.get("title", "")
        for m in ev.get("markets", []):
            try:
                yes_ask  = float(m.get("yes_ask_dollars", 1))
                yes_bid  = float(m.get("yes_bid_dollars", 0))
                yes_size = float(m.get("yes_ask_size_fp", 0) or 0)
            except (TypeError, ValueError):
                continue
            if yes_ask >= 0.99 or yes_bid <= 0.01 or yes_ask <= 0.01:
                continue
            sub = m.get("yes_sub_title", "").strip()
            question = f"{sub} — {parent_title}" if sub else parent_title
            candidates.append({
                "event_ticker": ev["event_ticker"],
                "ticker":       m.get("ticker", ""),
                "title":        question,
                "yes_ask":      round(yes_ask, 4),
                "yes_bid":      round(yes_bid, 4),
                "no_ask":       round(1.0 - yes_bid, 4),
                "yes_size":     yes_size,
                "volume":       float(m.get("volume_fp", 0) or 0),
                "is_mece":      True,
            })

    logger.info(f"Kalshi MECE candidates: {len(candidates)} individual markets")
    return candidates


# ─── Polymarket ──────────────────────────────────────────────────────────────

def _parse_field(val) -> list:
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except Exception:
        return []


async def fetch_poly_binary_markets(client: httpx.AsyncClient) -> list[dict]:
    """
    Page through Polymarket /markets. Keep only active binary (2-outcome, non-neg-risk)
    markets with snapshot prices in the 2%–98% range.
    """
    markets: list[dict] = []
    offset = 0

    for page in range(POLY_MAX_PAGES):
        try:
            r = await client.get(
                f"{GAMMA_API}/markets",
                params={"active": "true", "closed": "false", "limit": 200, "offset": offset},
                timeout=20,
            )
            r.raise_for_status()
            page_data = r.json()
        except Exception as e:
            logger.warning(f"Polymarket markets page {page + 1} failed: {e}")
            break
        if not isinstance(page_data, list) or not page_data:
            break

        for m in page_data:
            outcomes = _parse_field(m.get("outcomes", "[]"))
            if len(outcomes) != 2:
                continue
            prices = _parse_field(m.get("outcomePrices", "[]"))
            if len(prices) < 2:
                continue
            try:
                yes_snap = float(prices[0])
                no_snap  = float(prices[1])
            except (TypeError, ValueError):
                continue
            if yes_snap >= 0.98 or yes_snap <= 0.02:
                continue
            token_ids = _parse_field(m.get("clobTokenIds", "[]"))
            if len(token_ids) < 2:
                continue
            markets.append({
                "market_id":        str(m.get("id", "")),
                "question":         m.get("question", ""),
                "yes_token":        str(token_ids[0]),
                "no_token":         str(token_ids[1]),
                "yes_snap":         yes_snap,
                "no_snap":          no_snap,
                "volume":           float(m.get("volumeNum", 0) or 0),
                "end_date":         m.get("endDate", ""),
            })

        offset += len(page_data)
        if len(page_data) < 100:   # Polymarket caps pages at 100
            break

    logger.info(f"Polymarket: {len(markets)} active binary markets found")
    return markets


# ─── CLOB price fetch ─────────────────────────────────────────────────────────

async def _fetch_book(client: httpx.AsyncClient, token_id: str) -> dict:
    try:
        r = await client.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=5)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _book_depth(levels: list, price_cap: float, budget: float) -> tuple[float, float]:
    """
    Walk an order book (list of {price, size} levels sorted best-first) and
    compute the max contracts fillable within `budget` at prices <= `price_cap`.
    Returns (max_contracts, avg_fill_price). Returns (0, 0) if no liquidity.
    """
    remaining = budget
    total_contracts = 0.0
    total_cost = 0.0
    for level in levels:
        price = float(level["price"])
        if price > price_cap:
            break
        size = float(level["size"])
        affordable = remaining / price
        fill = min(size, affordable)
        total_contracts += fill
        total_cost += fill * price
        remaining -= fill * price
        if remaining < 0.001:
            break
    avg = (total_cost / total_contracts) if total_contracts > 0 else 0.0
    return total_contracts, round(avg, 6)


async def get_clob_prices(
    client: httpx.AsyncClient,
    yes_token: str,
    no_token: str,
    budget: float = 50.0,
) -> dict:
    """
    Fetch CLOB order books for both tokens. Returns a dict with:
      yes_ask, yes_bid, no_ask, no_bid  — best prices (None = no liquidity)
      yes_depth, no_depth               — max contracts within budget
      clob_verified                     — False if we had to fall back to snapshots
    Never falls back to snapshot prices: if there is no book, prices stay None
    and clob_verified=False. Callers must check this flag before trusting results.
    """
    yes_book, no_book = await asyncio.gather(
        _fetch_book(client, yes_token),
        _fetch_book(client, no_token),
    )
    yes_asks = yes_book.get("asks", [])
    yes_bids = yes_book.get("bids", [])
    no_asks  = no_book.get("asks", [])
    no_bids  = no_book.get("bids", [])

    yes_ask = float(yes_asks[0]["price"]) if yes_asks else None
    yes_bid = float(yes_bids[0]["price"]) if yes_bids else None
    no_ask  = float(no_asks[0]["price"])  if no_asks  else None
    no_bid  = float(no_bids[0]["price"])  if no_bids  else None

    # One-sided book: derive missing price from the other side if both missing
    # (this is legitimate — a market might only show one side when very skewed)
    if no_ask is None and no_bid is None and yes_bid is not None:
        no_ask = round(1.0 - yes_bid, 4)
    if yes_ask is None and yes_bid is None and no_bid is not None:
        yes_ask = round(1.0 - no_bid, 4)

    clob_verified = yes_ask is not None and no_ask is not None

    yes_depth, _ = _book_depth(yes_asks, 0.99, budget)
    no_depth,  _ = _book_depth(no_asks,  0.99, budget)

    return {
        "yes_ask":       yes_ask,
        "yes_bid":       yes_bid,
        "no_ask":        no_ask,
        "no_bid":        no_bid,
        "yes_depth":     yes_depth,
        "no_depth":      no_depth,
        "clob_verified": clob_verified,
    }


# ─── Matching ─────────────────────────────────────────────────────────────────

def _best_matches(
    kalshi: list[dict], poly: list[dict]
) -> list[tuple[dict, dict, float]]:
    """
    For each Kalshi binary market find the best-scoring Polymarket market.
    Returns sorted list of (km, pm, score) where score >= MIN_MATCH_SCORE.
    """
    pairs: list[tuple[dict, dict, float]] = []
    for km in kalshi:
        best_score = 0.0
        best_pm: Optional[dict] = None
        k_title = km.get("title", "")
        for pm in poly:
            sc, _, _ = match_score(k_title, pm.get("question", ""))
            if sc > best_score:
                best_score = sc
                best_pm = pm
        if best_score >= MIN_MATCH_SCORE and best_pm is not None:
            pairs.append((km, best_pm, round(best_score, 3)))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


# ─── Main scan ────────────────────────────────────────────────────────────────

async def scan_binary_arb(
    client: httpx.AsyncClient,
    min_profit: float = MIN_PROFIT,
    mece_opps: Optional[list] = None,   # pass from existing scan to avoid double-fetching
    max_position_usdc: float = 50.0,
) -> dict:
    """
    Full binary cross-platform scan.

    Pass `mece_opps` (output of KalshiClient.scan_mece_opportunities) to reuse
    an already-fetched MECE scan rather than running a second one.

    Returns:
      {
        "opportunities": [BinaryArb],        confirmed arbs (clob_verified only)
        "candidates":    [(km, pm, score)],  all matched pairs for inspection
        "kalshi_binary": int,
        "kalshi_mece":   int,
        "kalshi_count":  int,
        "poly_count":    int,
      }
    """
    # Fetch Polymarket first so we can use its questions to pre-filter Kalshi events.
    # MECE candidates are derived from the provided mece_opps (no extra API call).
    if mece_opps is None:
        from kalshi.kalshi_client import KalshiClient
        mece_opps = await KalshiClient().scan_mece_opportunities(client, min_profit=-1.0)

    poly_mkts = await fetch_poly_binary_markets(client)
    poly_questions = [pm["question"] for pm in poly_mkts]

    kalshi_binary = await fetch_kalshi_binary_markets(client, poly_questions=poly_questions)
    kalshi_mece = build_kalshi_mece_candidates(mece_opps)
    kalshi_mkts = kalshi_binary + kalshi_mece

    pairs = _best_matches(kalshi_mkts, poly_mkts)
    logger.info(f"Matched {len(pairs)} pairs (score >= {MIN_MATCH_SCORE})")

    sem = asyncio.Semaphore(CLOB_SEMAPHORE)

    async def check_pair(km: dict, pm: dict, score: float) -> Optional[BinaryArb]:
        async with sem:
            # Fast snapshot pre-filter — skip CLOB call if clearly unprofitable
            snap_dir1 = km["yes_ask"] + pm["no_snap"]
            snap_dir2 = pm["yes_snap"] + km["no_ask"]
            best_snap = max(1 - snap_dir1, 1 - snap_dir2)
            if best_snap < min_profit - SNAPSHOT_BUFFER:
                return None

            clob = await get_clob_prices(
                client, pm["yes_token"], pm["no_token"], budget=max_position_usdc
            )

            # If CLOB has no live book on either side, skip — snapshot prices
            # are last-trade (not executable asks) and would create phantom arbs.
            if not clob["clob_verified"]:
                return None

            p_yes = clob["yes_ask"]
            p_no  = clob["no_ask"]
            k_yes = km["yes_ask"]
            k_no  = km["no_ask"]

            profit_dir1 = 1.0 - k_yes - p_no   # buy Kalshi YES + Poly NO
            profit_dir2 = 1.0 - p_yes - k_no   # buy Poly YES + Kalshi NO

            best = max(profit_dir1, profit_dir2)
            if best < min_profit:
                return None

            if profit_dir1 >= profit_dir2:
                direction  = "buy_kalshi_yes_poly_no"
                k_leg, p_leg, profit = k_yes, p_no, profit_dir1
                # Limiting leg: Kalshi YES available size vs Poly NO book depth
                k_size  = km.get("yes_size", 0)
                p_depth = clob["no_depth"]
            else:
                direction  = "buy_poly_yes_kalshi_no"
                k_leg, p_leg, profit = k_no, p_yes, profit_dir2
                k_size  = km.get("yes_size", 0)   # Kalshi NO size ≈ YES size (same market)
                p_depth = clob["yes_depth"]

            # Max contracts: limited by budget, Kalshi available size, Poly book depth
            budget_contracts = max_position_usdc / (k_leg + p_leg) if (k_leg + p_leg) > 0 else 0
            max_contracts    = min(budget_contracts, k_size, p_depth) if k_size > 0 and p_depth > 0 \
                               else min(budget_contracts, max(k_size, p_depth))
            max_contracts    = max(round(max_contracts, 2), 0)

            return BinaryArb(
                kalshi_ticker  = km["ticker"],
                kalshi_title   = km["title"],
                poly_market_id = pm["market_id"],
                poly_question  = pm["question"],
                score          = score,
                direction      = direction,
                kalshi_leg     = round(k_leg, 4),
                poly_leg       = round(p_leg, 4),
                profit_pct     = round(profit, 4),
                max_contracts  = max_contracts,
                max_profit_usd = round(profit * max_contracts, 2),
                clob_verified  = True,
                k_yes_ask      = k_yes,
                k_no_ask       = k_no,
                p_yes_ask      = round(p_yes, 4),
                p_no_ask       = round(p_no, 4),
            )

    results = await asyncio.gather(*[check_pair(km, pm, sc) for km, pm, sc in pairs])
    opportunities = sorted(
        [r for r in results if r is not None],
        key=lambda x: x.profit_pct, reverse=True
    )

    return {
        "opportunities":    opportunities,
        "candidates":       pairs,
        "kalshi_binary":    len(kalshi_binary),
        "kalshi_mece":      len(kalshi_mece),
        "kalshi_count":     len(kalshi_mkts),
        "poly_count":       len(poly_mkts),
    }


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_binary_report(results: dict) -> None:
    opps  = results["opportunities"]
    pairs = results["candidates"]

    print(f"\n  {'─'*58}")
    print(f"  BINARY CROSS-PLATFORM ARB — Kalshi vs Polymarket")
    print(f"  Kalshi binary markets  : {results['kalshi_binary']}")
    print(f"  Kalshi MECE candidates : {results['kalshi_mece']}")
    print(f"  Polymarket binary mkts : {results['poly_count']}")
    print(f"  Total Kalshi pool      : {results['kalshi_count']}")
    print(f"  Matched pairs          : {len(pairs)}")
    print(f"  {'─'*58}")

    if opps:
        print(f"\n  *** {len(opps)} LIVE ARB OPPORTUNITY/IES ***\n")
        for i, arb in enumerate(opps, 1):
            side = ("Buy Kalshi YES + Poly NO"
                    if arb.direction == "buy_kalshi_yes_poly_no"
                    else "Buy Poly YES + Kalshi NO")
            print(f"  #{i}  {arb.profit_pct * 100:.2f}% profit  —  {side}")
            print(f"       Kalshi : {arb.kalshi_title[:58]}")
            print(f"       Poly   : {arb.poly_question[:58]}")
            print(f"       Match  : {arb.score:.2f}")
            print(f"       Kalshi   YES=${arb.k_yes_ask:.3f}  NO=${arb.k_no_ask:.3f}")
            print(f"       Poly     YES=${arb.p_yes_ask:.3f}  NO=${arb.p_no_ask:.3f}")
            print(f"       Cost/contract : ${arb.kalshi_leg + arb.poly_leg:.4f}")
            print(f"       Profit/contract: ${arb.profit_pct:.4f}")
            print()
    else:
        print(f"\n  No profitable opportunities right now.")

    # Always show top matched pairs so users can inspect and add to pairs.json
    if pairs:
        show = min(8, len(pairs))
        print(f"\n  Top {show} matched pairs (closest to arb):")
        print(f"  {'─'*58}")
        for km, pm, score in pairs[:show]:
            gap1 = 1.0 - km["yes_ask"] - pm["no_snap"]
            gap2 = 1.0 - pm["yes_snap"] - km["no_ask"]
            best_gap = max(gap1, gap2)
            dir_label = "K_YES+P_NO" if gap1 >= gap2 else "P_YES+K_NO"
            flag = "  ← PROFIT" if best_gap >= MIN_PROFIT else ""
            print(f"\n  [{score:.2f}] {dir_label}  gap={best_gap*100:+.1f}%{flag}")
            print(f"    Kalshi : {km['title'][:56]}")
            print(f"    Poly   : {pm['question'][:56]}")
            print(f"    K YES=${km['yes_ask']:.3f} NO=${km['no_ask']:.3f}  "
                  f"P YES≈${pm['yes_snap']:.3f} NO≈${pm['no_snap']:.3f}")
