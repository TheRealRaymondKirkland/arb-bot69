"""
Kalshi weather scanner — NO-only ensemble strategy.

Core insight from live trading analysis:
  - buy_yes (specific bucket wins) had 13% win rate — structurally bad because
    multiple YES bets per event and only one bucket can pay out.
  - buy_no (specific bucket DOESN'T win) had 70% win rate but bad economics
    because NO contracts priced at 88-90¢ only pay $1 back.

New strategy — only take NO bets when ALL of these are true:
  1. Ensemble probability for the bucket < 5% (≤ 2/39 members land there)
  2. Bucket center is ≥ 8°F from ensemble mean (model error can't bridge the gap)
  3. YES is bid at ≥ 20¢ (so we pay ≤ 80¢ for NO, payout ratio is worthwhile)
  4. Only ONE bet per city/metric/date (pick the highest-payout opportunity)
  5. Market closes within 36 hours (short-term forecasts are most accurate)

Economics with 90%+ win rate and NO_cost ≤ 0.80:
  EV = 0.90 × (1 - 0.80) × 0.93 - 0.10 × 0.80
     = 0.90 × 0.186 - 0.10 × 0.80
     = 0.167 - 0.080 = +$0.087 per contract (positive EV)

With 5 contracts per position: ~$0.43 expected profit on ~$4 deployed = ~10% per trade.
"""
import asyncio
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from typing import List, Optional, Tuple

import httpx

from weather.noaa_client import (
    get_daily_high as noaa_daily_high,
    get_daily_low  as noaa_daily_low,
    CITIES,
)
import weather.openmeteo_client as openmeteo

logger = logging.getLogger(__name__)

# ── Strategy parameters ───────────────────────────────────────────────────────
ENSEMBLE_MAX_PROB  = 0.05   # bucket must be < 5% likely per ensemble (≤ 2/39 members)
MIN_DISTANCE_F     = 8.0    # ensemble mean must be ≥ 8°F from bucket center
MIN_YES_BID        = 0.20   # YES must be bid ≥ 20¢ (so NO costs ≤ 80¢)
MAX_YES_ASK        = 0.50   # don't buy NO when YES > 50¢ (too risky)
MAX_HOURS_TO_CLOSE = 36     # only trade markets closing within 36 hours
MAX_CONTRACTS      = 5      # max 5 contracts per position (thin books)
MAX_DEPLOY         = 4.0    # max $4 deployed per position
MIN_PROFIT         = 0.10   # min total expected profit per position
KALSHI_TAKER_FEE   = 0.07   # 7% of price paid

# ── Known weather series ──────────────────────────────────────────────────────
SERIES_MAP: dict[str, tuple[str, str]] = {
    "KXHIGHTSEA":  ("seattle",      "high"),
    "KXLOWSEA":    ("seattle",      "low"),
    "KXHIGHHOU":   ("houston",      "high"),
    "KXHIGHOU":    ("houston",      "high"),
    "KXLOWHOU":    ("houston",      "low"),
    "KXHIGHNYD":   ("new york",     "high"),
    "KXHIGHNYC":   ("new york",     "high"),
    "KXHIGHNY":    ("new york",     "high"),
    "KXLOWNYC":    ("new york",     "low"),
    "KXLOWNY":     ("new york",     "low"),
    "KXLOWTAUS":   ("austin",       "low"),
    "KXHIGHAUST":  ("austin",       "high"),
    "KXHIGHAUS":   ("austin",       "high"),
    "KXLOWAUS":    ("austin",       "low"),
    "KXLOWTSATX":  ("san antonio",  "low"),
    "KXHIGHSAT":   ("san antonio",  "high"),
    "KXLOWSAT":    ("san antonio",  "low"),
    "KXLOWCHI":    ("chicago",      "low"),
    "KXHIGHCHI":   ("chicago",      "high"),
    "KXHIGHLAX":   ("los angeles",  "high"),
    "KXLOWLAX":    ("los angeles",  "low"),
    "KXHIGHDAL":   ("dallas",       "high"),
    "KXLOWDAL":    ("dallas",       "low"),
    "KXHIGHMIA":   ("miami",        "high"),
    "KXLOWMIA":    ("miami",        "low"),
    "KXHIGHPHX":   ("phoenix",      "high"),
    "KXLOWPHX":    ("phoenix",      "low"),
    "KXHIGHDEN":   ("denver",       "high"),
    "KXLOWDEN":    ("denver",       "low"),
    "KXHIGHATL":   ("atlanta",      "high"),
    "KXLOWATL":    ("atlanta",      "low"),
}

_SUFFIX_TO_CITY: dict[str, str] = {
    "SEA": "seattle",  "TSE": "seattle",
    "HOU": "houston",  "THOU": "houston", "IGHOU": "houston",
    "NYC": "new york", "NYD": "new york", "NY": "new york",
    "AUS": "austin",   "TAUS": "austin",  "AUST": "austin",
    "SATX": "san antonio", "SAT": "san antonio",
    "CHI": "chicago",
    "LAX": "los angeles",
    "DAL": "dallas",
    "MIA": "miami",
    "PHX": "phoenix",
    "DEN": "denver",
    "ATL": "atlanta",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slippage(n: int) -> float:
    if n <= 5:  return 0.000
    if n <= 15: return 0.005
    return 0.010


def _parse_range(title: str) -> Tuple[Optional[float], Optional[float]]:
    t = title.strip()
    m = re.match(r"(\d+(?:\.\d+)?)[°\s]*to[°\s]*(\d+(?:\.\d+)?)", t, re.I)
    if m: return float(m.group(1)), float(m.group(2))
    m = re.match(r"(\d+(?:\.\d+)?)[°\s]*or above", t, re.I)
    if m: return float(m.group(1)), None
    m = re.match(r"(\d+(?:\.\d+)?)[°\s]*or below", t, re.I)
    if m: return None, float(m.group(1))
    m = re.match(r"above[°\s]*(\d+(?:\.\d+)?)", t, re.I)
    if m: return float(m.group(1)), None
    m = re.match(r"below[°\s]*(\d+(?:\.\d+)?)", t, re.I)
    if m: return None, float(m.group(1))
    return None, None


def _bucket_center(lo: Optional[float], hi: Optional[float],
                   ensemble_mean: float) -> float:
    """Best estimate of bucket center for distance computation."""
    if lo is not None and hi is not None:
        return (lo + hi) / 2
    if lo is None:   # "below X"
        return hi - 5.0
    return lo + 5.0  # "above X"


def _parse_event_date(event_ticker: str) -> Optional[date]:
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", event_ticker)
    if not m:
        return None
    try:
        return datetime.strptime(f"20{m.group(1)} {m.group(2)} {m.group(3)}", "%Y %b %d").date()
    except ValueError:
        return None


# ── Dynamic series discovery ──────────────────────────────────────────────────

async def _discover_series(client: httpx.AsyncClient) -> dict[str, tuple[str, str]]:
    from kalshi.kalshi_client import _get
    discovered: dict[str, tuple[str, str]] = {}
    try:
        data = await _get(client, "/series", {"limit": 200})
        for s in data.get("series", []):
            ticker = s.get("ticker", "")
            if ticker.startswith("KXHIGH"):
                metric, suffix = "high", ticker[6:]
            elif ticker.startswith("KXLOW"):
                metric, suffix = "low",  ticker[5:]
            else:
                continue
            city = _SUFFIX_TO_CITY.get(suffix)
            if city:
                discovered[ticker] = (city, metric)
    except Exception as e:
        logger.debug(f"Series discovery failed: {e}")
    return {**discovered, **SERIES_MAP}


# ── Scanner dataclass ─────────────────────────────────────────────────────────

@dataclass
class WeatherOpportunity:
    event_ticker:    str
    ticker:          str
    city:            str
    metric:          str
    target_date:     str
    leg_title:       str
    low_f:           Optional[float]
    high_f:          Optional[float]
    kalshi_ask:      float
    kalshi_bid:      float
    forecast:        float          # ensemble mean for display
    prob:            float          # ensemble bucket probability
    edge:            float          # |kalshi_bid - prob| (how wrong the market is)
    net_profit:      float          # total expected profit (all contracts)
    action:          str            # always "buy_no"
    days_to_close:   float
    contracts:       int   = field(default=1)
    deploy_usd:      float = field(default=0.0)
    distance_f:      float = field(default=0.0)  # °F between bucket and ensemble mean

    @property
    def noaa_forecast(self) -> float: return self.forecast
    @property
    def noaa_prob(self)     -> float: return self.prob


# ── Main scanner ──────────────────────────────────────────────────────────────

_SEM = asyncio.Semaphore(8)


async def scan_weather_opportunities(
    client: httpx.AsyncClient,
    max_days: float = 1.5,
    portfolio_balance: float = 1000.0,
) -> dict:
    """
    Scan Kalshi weather markets for NO-bet opportunities where the ensemble
    says a bucket is nearly impossible but the market still prices it ≥ 20¢.

    Returns dict with 'opportunities', 'all_pairs', 'event_count', 'ensemble_pct'.
    """
    from kalshi.kalshi_client import _get

    series_map = await _discover_series(client)

    # ── Step 1: fetch all active events ──────────────────────────────────────
    weather_events: list[tuple[str, str, str]] = []

    async def fetch_events(prefix: str, city_metric: tuple):
        city, metric = city_metric
        async with _SEM:
            try:
                data = await _get(client, "/events", {
                    "series_ticker": prefix, "status": "open", "limit": 10
                })
                for e in data.get("events", []):
                    weather_events.append((e["event_ticker"], city, metric))
            except Exception:
                pass

    await asyncio.gather(*[fetch_events(pfx, cm) for pfx, cm in series_map.items()])
    logger.info(f"Weather scanner: {len(weather_events)} events from {len(series_map)} series")

    now_utc = datetime.now(timezone.utc)

    # ── Step 2: warm ensemble cache for relevant cities ───────────────────────
    active_cities = {city for _, city, _ in weather_events}
    await asyncio.gather(
        *[openmeteo._fetch_ensemble(c) for c in active_cities],
        return_exceptions=True,
    )

    # ── Step 3: process each event ────────────────────────────────────────────
    opportunities: list[WeatherOpportunity] = []
    all_pairs:     list[dict]               = []

    # Track best NO opportunity per (city, metric, date) to avoid multi-betting
    best_per_slot: dict[tuple, WeatherOpportunity] = {}

    async def process_event(event_ticker: str, city: str, metric: str):
        d = _parse_event_date(event_ticker)
        if not d:
            return

        days_out = (datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc) - now_utc).days
        if days_out < -1 or days_out > max_days:
            return

        # Get ensemble members once for this city/metric/date
        members = await openmeteo.get_member_extremes(city, d, metric)
        if not members or len(members) < 10:
            return
        ensemble_mean = sum(members) / len(members)

        async with _SEM:
            try:
                data = await _get(client, "/markets", {
                    "event_ticker": event_ticker, "limit": 25, "status": "open"
                })
            except Exception:
                return

        markets = [m for m in data.get("markets", [])
                   if m.get("result") == "" and m.get("status") == "active"]
        if not markets:
            return

        try:
            close_dt  = datetime.fromisoformat(markets[0]["close_time"].replace("Z", "+00:00"))
            hours_left = (close_dt - now_utc).total_seconds() / 3600
        except Exception:
            hours_left = 12.0

        if hours_left <= 0 or hours_left > MAX_HOURS_TO_CLOSE:
            return

        days_left = hours_left / 24

        for m in markets:
            leg    = m.get("yes_sub_title", "")
            lo, hi = _parse_range(leg)
            if lo is None and hi is None:
                continue

            # Ensemble probability for this specific bucket
            prob = openmeteo.bucket_prob(members, lo, hi)

            # Filter 1: must be nearly impossible per ensemble
            if prob > ENSEMBLE_MAX_PROB:
                continue

            # Filter 2: bucket must be far from ensemble mean
            center   = _bucket_center(lo, hi, ensemble_mean)
            distance = abs(ensemble_mean - center)
            if distance < MIN_DISTANCE_F:
                continue

            k_ask = float(m.get("yes_ask_dollars", 1.0))
            k_bid = float(m.get("yes_bid_dollars", 0.0))

            # Filter 3: YES must be priced in the right range for profitable NO
            if k_bid < MIN_YES_BID or k_ask > MAX_YES_ASK:
                continue

            # NO bet economics
            k_no     = 1.0 - k_bid
            slip     = _slippage(MAX_CONTRACTS)
            eff_no   = min(k_no + slip, 0.99)
            profit_c = (1.0 - prob) - eff_no * (1.0 + KALSHI_TAKER_FEE)

            if profit_c <= 0:
                continue

            n_c      = max(1, min(MAX_CONTRACTS, int(MAX_DEPLOY / eff_no)))
            profit   = round(profit_c * n_c, 4)
            deploy   = round(n_c * eff_no, 4)

            if profit < MIN_PROFIT:
                continue

            # Edge = how wrong the market is (YES_bid vs actual prob)
            edge = k_bid - prob  # positive means market overprices the bucket

            pair = {
                "event_ticker":  event_ticker,
                "ticker":        m.get("ticker", ""),
                "city":          city.title(),
                "metric":        metric,
                "target_date":   str(d),
                "leg_title":     leg,
                "low_f":         lo,
                "high_f":        hi,
                "kalshi_ask":    round(k_ask, 4),
                "kalshi_bid":    round(k_bid, 4),
                "eff_price":     round(eff_no, 4),
                "noaa_forecast": round(ensemble_mean, 1),
                "noaa_prob":     round(prob, 4),
                "edge":          round(edge, 4),
                "net_profit":    round(profit, 4),
                "action":        "buy_no",
                "days_to_close": round(days_left, 2),
                "contracts":     n_c,
                "deploy_usd":    deploy,
                "distance_f":    round(distance, 1),
                "prob_src":      "ensemble",
            }
            all_pairs.append(pair)

            # Keep only the best NO opportunity per (city, metric, date) slot
            slot = (city, metric, str(d))
            opp  = WeatherOpportunity(
                event_ticker=event_ticker,
                ticker=m.get("ticker", ""),
                city=city,
                metric=metric,
                target_date=str(d),
                leg_title=leg,
                low_f=lo,
                high_f=hi,
                kalshi_ask=round(k_ask, 4),
                kalshi_bid=round(k_bid, 4),
                forecast=round(ensemble_mean, 1),
                prob=round(prob, 4),
                edge=round(edge, 4),
                net_profit=round(profit, 4),
                action="buy_no",
                days_to_close=round(days_left, 2),
                contracts=n_c,
                deploy_usd=deploy,
                distance_f=round(distance, 1),
            )
            existing = best_per_slot.get(slot)
            if existing is None or k_bid > existing.kalshi_bid:
                best_per_slot[slot] = opp

    await asyncio.gather(*[
        process_event(et, city, metric)
        for et, city, metric in weather_events
    ])

    # Build final opportunity list from best-per-slot
    opportunities = sorted(best_per_slot.values(), key=lambda x: x.kalshi_bid, reverse=True)
    all_pairs.sort(key=lambda x: x["edge"], reverse=True)

    unique_events = len({et for et, _, _ in weather_events})
    logger.info(
        f"Weather scan (NO-only): {len(all_pairs)} candidates → "
        f"{len(opportunities)} best-per-slot opportunities"
    )
    return {
        "opportunities": opportunities,
        "all_pairs":     all_pairs,
        "event_count":   unique_events,
        "ensemble_pct":  100,   # always ensemble now
    }
