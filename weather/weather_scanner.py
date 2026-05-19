"""
Kalshi weather market scanner — NOAA + Tomorrow.io ensemble vs Kalshi prices.

Edge: NOAA NWS forecasts are ~90% accurate for next-day temps.
      Tomorrow.io ML model (when API key set) provides an independent signal.
      Ensemble = 40% NOAA + 60% Tomorrow.io (or 100% NOAA if Tomorrow.io unavailable).
Kalshi weather markets are priced by retail traders who rarely check forecasts.
When ensemble probability for a bucket differs from Kalshi's price by > MIN_EDGE,
we have a tradeable opportunity.

Position sizing: Quarter-Kelly criterion capped at 10% of portfolio per trade.
  f* = (p*b - q) / b   (full Kelly fraction)
  deploy = min(f*/4 * balance, 0.10 * balance)
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

MIN_EDGE   = 0.12   # minimum |ensemble_prob - kalshi_price| to flag
SIGMA      = 3.0    # NWS 24h forecast std-dev in °F
MIN_PROFIT = 0.02   # min net profit after fees per contract

KALSHI_TAKER_FEE = 0.07

# ── Known weather series (hardcoded fallback + seed for dynamic discovery) ────
SERIES_MAP: dict[str, tuple[str, str]] = {
    # Seattle
    "KXHIGHTSEA":  ("seattle",      "high"),
    "KXLOWSEA":    ("seattle",      "low"),
    # Houston
    "KXHIGHHOU":   ("houston",      "high"),
    "KXHIGHOU":    ("houston",      "high"),
    "KXLOWHOU":    ("houston",      "low"),
    # New York
    "KXHIGHNYD":   ("new york",     "high"),
    "KXHIGHNYC":   ("new york",     "high"),
    "KXHIGHNY":    ("new york",     "high"),
    "KXLOWNYC":    ("new york",     "low"),
    "KXLOWNY":     ("new york",     "low"),
    # Austin
    "KXLOWTAUS":   ("austin",       "low"),
    "KXHIGHAUST":  ("austin",       "high"),
    "KXLOWAUS":    ("austin",       "low"),
    # San Antonio
    "KXLOWTSATX":  ("san antonio",  "low"),
    "KXHIGHSAT":   ("san antonio",  "high"),
    "KXLOWSAT":    ("san antonio",  "low"),
    # Chicago
    "KXLOWCHI":    ("chicago",      "low"),
    "KXHIGHCHI":   ("chicago",      "high"),
    # Los Angeles
    "KXHIGHLAX":   ("los angeles",  "high"),
    "KXLOWLAX":    ("los angeles",  "low"),
    # Dallas
    "KXHIGHDAL":   ("dallas",       "high"),
    "KXLOWDAL":    ("dallas",       "low"),
    # Miami
    "KXHIGHMIA":   ("miami",        "high"),
    "KXLOWMIA":    ("miami",        "low"),
    # Phoenix
    "KXHIGHPHX":   ("phoenix",      "high"),
    "KXLOWPHX":    ("phoenix",      "low"),
    # Denver
    "KXHIGHDEN":   ("denver",       "high"),
    "KXLOWDEN":    ("denver",       "low"),
    # Atlanta
    "KXHIGHATL":   ("atlanta",      "high"),
    "KXLOWATL":    ("atlanta",      "low"),
}

# Suffix → city mapping for dynamic discovery
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


# ── Probability math ──────────────────────────────────────────────────────────

def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _prob(forecast: float, low: Optional[float], high: Optional[float],
          sigma: float = SIGMA) -> float:
    lo = -1e9 if low  is None else low  - 0.5
    hi =  1e9 if high is None else high + 0.5
    return _phi((hi - forecast) / sigma) - _phi((lo - forecast) / sigma)


# ── Kelly criterion ───────────────────────────────────────────────────────────

def kelly_contracts(p: float, ask: float, balance: float,
                    max_pct: float = 0.10) -> int:
    """
    Quarter-Kelly position sizing. Returns integer contract count ≥ 1.
    p:       our probability estimate
    ask:     price per YES contract (0–1)
    balance: current virtual portfolio balance
    """
    if ask <= 0 or ask >= 1 or p <= ask:   # no edge or degenerate
        return 1
    b     = (1.0 - ask) / ask              # net profit per dollar risked
    f_full  = (p * b - (1.0 - p)) / b     # full Kelly fraction
    deploy  = min(f_full * 0.25 * balance, max_pct * balance)
    return max(1, int(deploy / ask))


# ── Ensemble forecast (display only) ─────────────────────────────────────────

async def _point_forecast(city: str, metric: str, d: date) -> Optional[float]:
    """
    Get a single point forecast for display. Tries Open-Meteo mean, then NOAA.
    NOT used for probability — that comes from get_ensemble_prob() below.
    """
    noaa_coro = noaa_daily_high(city, d) if metric == "high" else noaa_daily_low(city, d)
    om_coro   = openmeteo.get_daily_high(city, d) if metric == "high" else openmeteo.get_daily_low(city, d)
    noaa_f, om_f = await asyncio.gather(noaa_coro, om_coro)

    if om_f is not None and noaa_f is not None:
        return round(0.4 * noaa_f + 0.6 * om_f, 1)
    return om_f or noaa_f


async def get_ensemble_prob(
    city: str, d: date, metric: str,
    low_f: Optional[float], high_f: Optional[float],
) -> tuple[float, str]:
    """
    Compute P(metric in [low_f, high_f]) using Open-Meteo ensemble (40 members).
    Falls back to NOAA + Gaussian if ensemble unavailable.
    Returns (probability, source_label).
    """
    om_prob = await openmeteo.get_bucket_prob(city, d, metric, low_f, high_f)
    if om_prob is not None:
        return om_prob, "ensemble"
    # Fallback: NOAA point + Gaussian sigma=3°F
    noaa_f = await (noaa_daily_high(city, d) if metric == "high" else noaa_daily_low(city, d))
    if noaa_f is not None:
        return _prob(noaa_f, low_f, high_f), "noaa+gaussian"
    return 0.5, "unknown"


# ── Range parser ──────────────────────────────────────────────────────────────

def _parse_range(title: str) -> Tuple[Optional[float], Optional[float]]:
    t = title.strip()
    m = re.match(r"(\d+(?:\.\d+)?)[°\s]*to[°\s]*(\d+(?:\.\d+)?)", t, re.I)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"(\d+(?:\.\d+)?)[°\s]*or above", t, re.I)
    if m:
        return float(m.group(1)), None
    m = re.match(r"(\d+(?:\.\d+)?)[°\s]*or below", t, re.I)
    if m:
        return None, float(m.group(1))
    m = re.match(r"above[°\s]*(\d+(?:\.\d+)?)", t, re.I)
    if m:
        return float(m.group(1)), None
    m = re.match(r"below[°\s]*(\d+(?:\.\d+)?)", t, re.I)
    if m:
        return None, float(m.group(1))
    return None, None


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
    """
    Query Kalshi /series to find all KXHIGH*/KXLOW* series.
    Merges with SERIES_MAP (hardcoded takes priority).
    """
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
        if discovered:
            logger.info(f"Dynamic discovery: {len(discovered)} weather series found")
    except Exception as e:
        logger.debug(f"Series discovery failed: {e}")
    # Hardcoded takes priority
    return {**discovered, **SERIES_MAP}


# ── Scanner dataclass ─────────────────────────────────────────────────────────

@dataclass
class WeatherOpportunity:
    event_ticker:  str
    ticker:        str
    city:          str
    metric:        str
    target_date:   str
    leg_title:     str
    low_f:         Optional[float]
    high_f:        Optional[float]
    kalshi_ask:    float
    kalshi_bid:    float
    forecast:      float
    prob:          float
    edge:          float
    net_profit:    float
    action:        str
    days_to_close: float
    contracts:     int    = field(default=1)
    deploy_usd:    float  = field(default=0.0)
    tomorrow_on:   bool   = field(default=False)

    # back-compat alias
    @property
    def noaa_forecast(self) -> float:
        return self.forecast

    @property
    def noaa_prob(self) -> float:
        return self.prob


# ── Main scanner ──────────────────────────────────────────────────────────────

_SEM = asyncio.Semaphore(8)   # max 8 concurrent Kalshi API calls


async def scan_weather_opportunities(
    client: httpx.AsyncClient,
    min_edge: float = MIN_EDGE,
    max_days: float = 3.0,
    portfolio_balance: float = 1000.0,
) -> dict:
    """
    Scan all active Kalshi weather markets vs NOAA + Tomorrow.io ensemble.
    Returns dict with 'opportunities', 'all_pairs', 'event_count', 'tomorrow_on'.
    """
    from kalshi.kalshi_client import _get

    series_map = await _discover_series(client)

    # ── Step 1: fetch all active events in parallel ───────────────────────────
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

    await asyncio.gather(*[
        fetch_events(pfx, cm) for pfx, cm in series_map.items()
    ])

    logger.info(f"Weather scanner: {len(weather_events)} events from {len(series_map)} series")

    now_utc = datetime.now(timezone.utc)

    # ── Step 2: pre-warm Open-Meteo ensemble cache for all relevant cities ────
    active_cities = {city for _, city, _ in weather_events}
    await asyncio.gather(
        *[openmeteo._fetch_ensemble(c) for c in active_cities],
        return_exceptions=True,
    )

    # Point forecast cache for display only
    combos: set[tuple[str, str, date]] = set()
    for et, city, metric in weather_events:
        d = _parse_event_date(et)
        if not d:
            continue
        days_out = (datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc) - now_utc).days
        if 0 <= days_out <= max_days:
            combos.add((city, metric, d))

    point_cache: dict[tuple, Optional[float]] = {}

    async def do_point(city: str, metric: str, d: date):
        try:
            point_cache[(city, metric, d)] = await _point_forecast(city, metric, d)
        except Exception as e:
            logger.warning(f"Point forecast failed {city}/{metric}/{d}: {e}")

    await asyncio.gather(*[do_point(c, m, d) for c, m, d in combos])

    # ── Step 3: fetch market legs + compute edges in parallel ─────────────────
    opportunities: list[WeatherOpportunity] = []
    all_pairs:     list[dict]               = []

    async def process_event(event_ticker: str, city: str, metric: str):
        d = _parse_event_date(event_ticker)
        if not d:
            return
        days_out = (datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc) - now_utc).days
        if days_out < 0 or days_out > max_days:
            return

        forecast = point_cache.get((city, metric, d))   # for display only

        async with _SEM:
            try:
                data = await _get(client, "/markets", {
                    "event_ticker": event_ticker, "limit": 20, "status": "open"
                })
            except Exception:
                return

        markets = [m for m in data.get("markets", [])
                   if m.get("result") == "" and m.get("status") == "active"]
        if not markets:
            return

        try:
            close_dt  = datetime.fromisoformat(markets[0]["close_time"].replace("Z", "+00:00"))
            days_left = (close_dt - now_utc).total_seconds() / 86400
        except Exception:
            days_left = 1.0

        logger.debug(
            f"{event_ticker} | {city} {metric} {d} | "
            f"point={forecast}°F | {len(markets)} legs | {days_left:.1f}d"
        )

        # Compute ensemble probs for all legs in this event in parallel
        leg_data = []
        for m in markets:
            leg    = m.get("yes_sub_title", "")
            lo, hi = _parse_range(leg)
            if lo is None and hi is None:
                continue
            leg_data.append((m, leg, lo, hi))

        if not leg_data:
            return

        prob_results = await asyncio.gather(*[
            get_ensemble_prob(city, d, metric, lo, hi)
            for _, _, lo, hi in leg_data
        ])

        for (m, leg, lo, hi), (p, prob_src) in zip(leg_data, prob_results):
            k_ask = float(m.get("yes_ask_dollars", 1.0))
            k_bid = float(m.get("yes_bid_dollars", 0.0))
            edge  = p - k_ask

            if edge > 0:
                fee    = KALSHI_TAKER_FEE * k_ask * (1.0 - k_ask)
                profit = p * (1.0 - k_ask) - (1.0 - p) * k_ask - fee
                action = "buy_yes"
                n_c    = kelly_contracts(p, k_ask, portfolio_balance)
                deploy = round(n_c * k_ask, 4)
            else:
                k_no  = 1.0 - k_bid
                fee   = KALSHI_TAKER_FEE * k_no * (1.0 - k_no)
                profit = (1.0 - p) * (1.0 - k_no) - p * k_no - fee
                action = "buy_no"
                n_c   = kelly_contracts(1.0 - p, k_no, portfolio_balance)
                deploy = round(n_c * k_no, 4)

            display_forecast = forecast if forecast is not None else 0.0
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
                "noaa_forecast": display_forecast,
                "noaa_prob":     round(p, 4),
                "edge":          round(edge, 4),
                "net_profit":    round(profit, 4),
                "action":        action,
                "days_to_close": round(days_left, 2),
                "contracts":     n_c,
                "deploy_usd":    deploy,
                "prob_src":      prob_src,
            }
            all_pairs.append(pair)

            if abs(edge) >= min_edge and profit >= MIN_PROFIT:
                opportunities.append(WeatherOpportunity(
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
                    forecast=display_forecast,
                    prob=round(p, 4),
                    edge=round(edge, 4),
                    net_profit=round(profit, 4),
                    action=action,
                    days_to_close=round(days_left, 2),
                    contracts=n_c,
                    deploy_usd=deploy,
                ))

    await asyncio.gather(*[
        process_event(et, city, metric)
        for et, city, metric in weather_events
    ])

    opportunities.sort(key=lambda x: abs(x.edge), reverse=True)
    all_pairs.sort(key=lambda x: abs(x["edge"]), reverse=True)

    unique_events = len({et for et, _, _ in weather_events})
    ensemble_count = sum(1 for p in all_pairs if p.get("prob_src") == "ensemble")
    logger.info(
        f"Weather scan: {len(all_pairs)} legs · {len(opportunities)} edges "
        f"(≥{min_edge:.0%}) · ensemble={ensemble_count}/{len(all_pairs)} legs"
    )
    return {
        "opportunities": opportunities,
        "all_pairs":     all_pairs,
        "event_count":   unique_events,
        "ensemble_pct":  round(ensemble_count / max(len(all_pairs), 1) * 100),
    }
