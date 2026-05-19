"""
Cross-platform sports arb scanner: Kalshi vs Polymarket.

Monitors championship/playoff markets on both platforms and alerts when
prices diverge enough to arb. The two platforms update at different speeds
when news breaks (injuries, game results) — that's the window.

Arb logic (cross-platform binary):
  Buy YES team X on Polymarket at poly_yes_ask
  Buy NO team X on Kalshi   at kalshi_no_ask  (~1 - kalshi_yes_bid)
  Payout: always $1 (one side wins)
  Profit: 1 - poly_yes_ask - kalshi_no_ask - fees

Or the reverse:
  Buy YES team X on Kalshi   at kalshi_yes_ask
  Buy NO team X on Polymarket at poly_no_ask (~1 - poly_yes_bid)
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

KALSHI_TAKER_FEE  = 0.07   # 0.07 * price * (1-price)
POLY_TAKER_FEE    = 0.02   # flat 2% of trade value

# ── Known Kalshi championship events ─────────────────────────────────────────
KALSHI_SPORT_EVENTS = [
    "KXNBA-26",    # NBA Finals 2026
    "KXNHL-26",    # Stanley Cup 2026
    "KXMLB-26",    # World Series 2026
    "KXNFL-27",    # Super Bowl 2027
]

# ── Team name normalization ───────────────────────────────────────────────────
_ALIASES: Dict[str, str] = {
    # NBA
    "san antonio":       "spurs",
    "spurs":             "spurs",
    "oklahoma city":     "thunder",
    "thunder":           "thunder",
    "new york":          "knicks",
    "knicks":            "knicks",
    "cleveland":         "cavaliers",
    "cavaliers":         "cavaliers",
    "cavs":              "cavaliers",
    "indiana":           "pacers",
    "pacers":            "pacers",
    "miami":             "heat",
    "heat":              "heat",
    "boston":            "celtics",
    "celtics":           "celtics",
    "minnesota":         "timberwolves",
    "timberwolves":      "timberwolves",
    "wolves":            "timberwolves",
    "golden state":      "warriors",
    "warriors":          "warriors",
    "denver":            "nuggets",
    "nuggets":           "nuggets",
    "los angeles lakers":"lakers",
    "lakers":            "lakers",
    "los angeles clippers":"clippers",
    "clippers":          "clippers",
    # NHL
    "colorado":          "avalanche",
    "avalanche":         "avalanche",
    "carolina":          "hurricanes",
    "hurricanes":        "hurricanes",
    "vegas":             "golden knights",
    "golden knights":    "golden knights",
    "montreal":          "canadiens",
    "canadiens":         "canadiens",
    "montréal":          "canadiens",
    "edmonton":          "oilers",
    "oilers":            "oilers",
    "florida":           "panthers",
    "panthers":          "panthers",
    "dallas":            "stars",
    "stars":             "stars",
    "new jersey":        "devils",
    "devils":            "devils",
    # MLB
    "new york yankees":  "yankees",
    "yankees":           "yankees",
    "new york mets":     "mets",
    "mets":              "mets",
    "los angeles dodgers":"dodgers",
    "dodgers":           "dodgers",
    "atlanta":           "braves",
    "braves":            "braves",
    "houston":           "astros",
    "astros":            "astros",
    "philadelphia":      "phillies",
    "phillies":          "phillies",
    "chicago cubs":      "cubs",
    "cubs":              "cubs",
    "chicago white sox": "white sox",
    "white sox":         "white sox",
    "san diego":         "padres",
    "padres":            "padres",
    "seattle":           "mariners",
    "mariners":          "mariners",
    "baltimore":         "orioles",
    "orioles":           "orioles",
}


def _normalize(name: str) -> str:
    name = name.lower().strip()
    for alias, canonical in _ALIASES.items():
        if alias in name:
            return canonical
    return name


@dataclass
class SportArb:
    team:           str
    sport:          str
    direction:      str          # "buy_poly_yes" or "buy_kalshi_yes"
    poly_yes_ask:   float
    kalshi_yes_ask: float
    kalshi_yes_bid: float
    profit_pct:     float        # net after fees
    gross_profit:   float
    kalshi_ticker:  str
    poly_market_id: str
    poly_question:  str


@dataclass
class SportPair:
    team:           str
    sport:          str
    kalshi_yes_ask: float
    kalshi_yes_bid: float
    poly_yes_ask:   float
    poly_yes_bid:   float
    gap:            float        # best gross spread (may be negative)
    direction:      str
    kalshi_ticker:  str
    poly_market_id: str
    poly_question:  str


async def _fetch_kalshi_sport_markets(client: httpx.AsyncClient) -> List[dict]:
    """Fetch all known Kalshi championship markets."""
    from kalshi.kalshi_client import _get
    results = []
    for event_ticker in KALSHI_SPORT_EVENTS:
        try:
            data = await _get(client, "/markets", {
                "event_ticker": event_ticker, "limit": 50, "status": "open"
            })
            for m in data.get("markets", []):
                if m.get("result") == "" and m.get("status") == "active":
                    m["_event"] = event_ticker
                    results.append(m)
        except Exception as e:
            logger.debug(f"Kalshi sport event {event_ticker}: {e}")
    return results


async def _fetch_poly_sport_markets(client: httpx.AsyncClient) -> List[dict]:
    """Fetch Polymarket sports markets with price data."""
    sports_kw = [
        "nba finals", "stanley cup", "world series", "super bowl",
        "nba champion", "nhl champion", "mlb champion", "nfl champion",
        "win the 2026", "win the 2027", "western conference", "eastern conference",
        "conference final", "playoffs",
    ]
    all_markets = []
    try:
        offset = 0
        async with httpx.AsyncClient() as c:
            while True:
                r = await c.get(
                    f"https://gamma-api.polymarket.com/markets?limit=100&active=true&offset={offset}",
                    timeout=15
                )
                batch = r.json()
                if not batch:
                    break
                all_markets.extend(batch)
                offset += 100
                if len(batch) < 100:
                    break
    except Exception as e:
        logger.warning(f"Polymarket fetch error: {e}")

    result = []
    for m in all_markets:
        q = m.get("question", "").lower()
        if any(kw in q for kw in sports_kw):
            # Parse best bid/ask from outcomePrices and tokens
            try:
                prices = m.get("outcomePrices", "[]")
                if isinstance(prices, str):
                    import json
                    prices = json.loads(prices)
                tokens = m.get("tokens", [])
                yes_tok = next((t for t in tokens if t.get("outcome","").lower() == "yes"), None)
                if yes_tok:
                    m["_yes_price"] = float(yes_tok.get("price", prices[0] if prices else 0.5))
                elif prices:
                    m["_yes_price"] = float(prices[0])
                else:
                    continue
                # bestAsk/bestBid from CLOB if available
                m["_yes_ask"] = float(m.get("bestAsk") or m["_yes_price"])
                m["_yes_bid"] = float(m.get("bestBid") or m["_yes_price"])
                result.append(m)
            except Exception:
                continue
    return result


def _kalshi_fee(price: float) -> float:
    return KALSHI_TAKER_FEE * price * (1.0 - price)


def _poly_fee(price: float) -> float:
    return POLY_TAKER_FEE * price


async def scan_sports_arb(
    client: httpx.AsyncClient,
    min_profit: float = 0.01,
) -> dict:
    """
    Scan Kalshi + Polymarket championship markets for cross-platform arb.
    Returns opportunities (profit > min_profit) and all matched pairs for display.
    """
    k_markets, p_markets = await asyncio.gather(
        _fetch_kalshi_sport_markets(client),
        _fetch_poly_sport_markets(client),
    )

    logger.info(f"Sports scan: {len(k_markets)} Kalshi markets, {len(p_markets)} Polymarket markets")

    # Build Polymarket lookup: normalized_team → market
    poly_by_team: Dict[str, dict] = {}
    for pm in p_markets:
        q = pm.get("question", "")
        norm = _normalize(q)
        poly_by_team[norm] = pm
        # Also index by individual team words
        for word in norm.split():
            if len(word) > 3 and word not in ("will", "the", "win", "2026", "2027", "nba", "nhl", "mlb", "nfl"):
                if word not in poly_by_team:
                    poly_by_team[word] = pm

    pairs: List[SportPair] = []
    opportunities: List[SportArb] = []

    for km in k_markets:
        sub_title = km.get("yes_sub_title", km.get("title", ""))
        team_norm = _normalize(sub_title)
        event = km.get("_event", "")
        sport = (
            "NBA" if "NBA" in event else
            "NHL" if "NHL" in event else
            "MLB" if "MLB" in event else
            "NFL" if "NFL" in event else "Sport"
        )

        # Match to Polymarket
        pm = poly_by_team.get(team_norm)
        if pm is None:
            # Try partial match
            for alias, canonical in _ALIASES.items():
                if canonical == team_norm or alias in team_norm:
                    pm = poly_by_team.get(canonical)
                    if pm:
                        break
        if pm is None:
            logger.debug(f"No Polymarket match for Kalshi team: {sub_title} ({team_norm})")
            continue

        k_yes_ask = float(km.get("yes_ask_dollars", 1.0))
        k_yes_bid = float(km.get("yes_bid_dollars", k_yes_ask - 0.02))
        k_no_ask  = round(1.0 - k_yes_bid, 4)   # cost to buy NO on Kalshi

        p_yes_ask = float(pm["_yes_ask"])
        p_yes_bid = float(pm["_yes_bid"])
        p_no_ask  = round(1.0 - p_yes_bid, 4)   # cost to buy NO on Polymarket

        # Direction A: Buy YES Poly + NO Kalshi
        cost_a    = p_yes_ask + k_no_ask
        fee_a     = _poly_fee(p_yes_ask) + _kalshi_fee(k_no_ask)
        profit_a  = 1.0 - cost_a - fee_a

        # Direction B: Buy YES Kalshi + NO Poly
        cost_b    = k_yes_ask + p_no_ask
        fee_b     = _kalshi_fee(k_yes_ask) + _poly_fee(p_no_ask)
        profit_b  = 1.0 - cost_b - fee_b

        best_profit = max(profit_a, profit_b)
        direction   = "buy_poly_yes" if profit_a >= profit_b else "buy_kalshi_yes"
        gap         = max(1.0 - cost_a, 1.0 - cost_b)  # gross gap

        pair = SportPair(
            team=sub_title,
            sport=sport,
            kalshi_yes_ask=k_yes_ask,
            kalshi_yes_bid=k_yes_bid,
            poly_yes_ask=p_yes_ask,
            poly_yes_bid=p_yes_bid,
            gap=round(gap, 4),
            direction=direction,
            kalshi_ticker=km.get("ticker", ""),
            poly_market_id=pm.get("id", ""),
            poly_question=pm.get("question", ""),
        )
        pairs.append(pair)

        if best_profit >= min_profit:
            opportunities.append(SportArb(
                team=sub_title,
                sport=sport,
                direction=direction,
                poly_yes_ask=p_yes_ask,
                kalshi_yes_ask=k_yes_ask,
                kalshi_yes_bid=k_yes_bid,
                profit_pct=round(best_profit, 4),
                gross_profit=round(gap, 4),
                kalshi_ticker=km.get("ticker", ""),
                poly_market_id=pm.get("id", ""),
                poly_question=pm.get("question", ""),
            ))

    pairs.sort(key=lambda x: x.gap, reverse=True)
    opportunities.sort(key=lambda x: x.profit_pct, reverse=True)

    logger.info(
        f"Sports arb: {len(pairs)} matched pairs, {len(opportunities)} opportunities "
        f"(min_profit={min_profit:.1%})"
    )
    return {
        "pairs":         pairs,
        "opportunities": opportunities,
        "kalshi_count":  len(k_markets),
        "poly_count":    len(p_markets),
    }
