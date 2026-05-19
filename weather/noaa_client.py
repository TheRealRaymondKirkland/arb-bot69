"""
NOAA NWS forecast client — no API key required.

Fetches hourly forecasts and computes:
  - Daily high temperature for a city/date
  - Daily low temperature for a city/date
  - Max precipitation probability for a city/date

Caches gridpoints permanently (they never change) and forecasts for 30 min.
"""
import asyncio
import json
import logging
import time
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple
import httpx

logger = logging.getLogger(__name__)

CACHE_DIR   = Path(__file__).parent.parent / "data"
GRID_CACHE  = CACHE_DIR / "noaa_gridpoints.json"
FCST_CACHE  = CACHE_DIR / "noaa_forecasts.json"
FCST_TTL    = 1800   # 30 min — NOAA updates hourly

# Known cities: name → (lat, lon, tz_offset_hours)
CITIES = {
    "seattle":      (47.6062, -122.3321, -7),
    "austin":       (30.2672,  -97.7431, -5),
    "san antonio":  (29.4241,  -98.4936, -5),
    "houston":      (29.7604,  -95.3698, -5),
    "new york":     (40.7128,  -74.0060, -4),
    "nyc":          (40.7128,  -74.0060, -4),
    "los angeles":  (34.0522, -118.2437, -7),
    "chicago":      (41.8781,  -87.6298, -5),
    "dallas":       (32.7767,  -96.7970, -5),
    "miami":        (25.7617,  -80.1918, -4),
    "phoenix":      (33.4484, -112.0740, -7),
    "denver":       (39.7392, -104.9903, -6),
    "atlanta":      (33.7490,  -84.3880, -4),
}

HEADERS = {"User-Agent": "arb-bot/1.0 (weather scanner; contact@example.com)"}


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def _save_json(path: Path, data: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


async def _get_gridpoint(city: str) -> Optional[Tuple[str, int, int]]:
    """Return (wfo, x, y) for a city — cached permanently."""
    cache = _load_json(GRID_CACHE)
    if city in cache:
        return tuple(cache[city])

    if city not in CITIES:
        logger.warning(f"Unknown city: {city}")
        return None

    lat, lon, _ = CITIES[city]
    url = f"https://api.weather.gov/points/{lat},{lon}"
    try:
        async with httpx.AsyncClient(headers=HEADERS) as client:
            r = await client.get(url, timeout=10)
            r.raise_for_status()
            props = r.json()["properties"]
            wfo   = props["gridId"]
            gx    = props["gridX"]
            gy    = props["gridY"]
            cache[city] = [wfo, gx, gy]
            _save_json(GRID_CACHE, cache)
            return wfo, gx, gy
    except Exception as e:
        logger.warning(f"NOAA gridpoint lookup failed for {city}: {e}")
        return None


async def _fetch_hourly(city: str) -> list:
    """Return list of hourly forecast dicts for the next ~7 days."""
    cache = _load_json(FCST_CACHE)
    entry = cache.get(city, {})
    if entry and time.time() - entry.get("ts", 0) < FCST_TTL:
        return entry["periods"]

    gp = await _get_gridpoint(city)
    if not gp:
        return []

    wfo, gx, gy = gp
    url = f"https://api.weather.gov/gridpoints/{wfo}/{gx},{gy}/forecast/hourly"
    try:
        async with httpx.AsyncClient(headers=HEADERS) as client:
            r = await client.get(url, timeout=15)
            r.raise_for_status()
            periods = r.json()["properties"]["periods"]
            cache[city] = {"ts": time.time(), "periods": periods}
            _save_json(FCST_CACHE, cache)
            logger.info(f"NOAA forecast fetched for {city}: {len(periods)} hours")
            return periods
    except Exception as e:
        logger.warning(f"NOAA hourly fetch failed for {city}: {e}")
        return []


def _local_date_str(period: dict, tz_offset: int) -> str:
    """Return YYYY-MM-DD in local time for a forecast period."""
    try:
        dt = datetime.fromisoformat(period["startTime"].replace("Z", "+00:00"))
        local = dt + timedelta(hours=tz_offset)
        return local.strftime("%Y-%m-%d")
    except Exception:
        return ""


async def get_daily_high(city: str, target_date: Optional[date] = None) -> Optional[float]:
    """Return NOAA forecast daily high temp (°F) for a city on a given date."""
    city = city.lower().strip()
    tz_offset = CITIES.get(city, (0, 0, -5))[2]
    if target_date is None:
        now_local = datetime.now(timezone.utc) + timedelta(hours=tz_offset)
        target_date = now_local.date()

    target_str = target_date.strftime("%Y-%m-%d")
    periods = await _fetch_hourly(city)
    temps = [
        p["temperature"] for p in periods
        if _local_date_str(p, tz_offset) == target_str
        and p.get("isDaytime", True)
    ]
    if not temps:
        # Fall back to all hours if isDaytime filter returns nothing
        temps = [p["temperature"] for p in periods if _local_date_str(p, tz_offset) == target_str]

    return float(max(temps)) if temps else None


async def get_daily_low(city: str, target_date: Optional[date] = None) -> Optional[float]:
    """Return NOAA forecast daily low temp (°F) for a city on a given date."""
    city = city.lower().strip()
    tz_offset = CITIES.get(city, (0, 0, -5))[2]
    if target_date is None:
        now_local = datetime.now(timezone.utc) + timedelta(hours=tz_offset)
        target_date = now_local.date()

    target_str = target_date.strftime("%Y-%m-%d")
    periods = await _fetch_hourly(city)
    temps = [p["temperature"] for p in periods if _local_date_str(p, tz_offset) == target_str]
    return float(min(temps)) if temps else None


async def prefetch_all(cities: Optional[list] = None) -> None:
    """Warm the forecast cache for all (or specified) cities in parallel."""
    targets = cities or list(CITIES.keys())
    await asyncio.gather(*[_fetch_hourly(c) for c in targets], return_exceptions=True)


async def get_precip_probability(city: str, target_date: Optional[date] = None) -> Optional[float]:
    """Return max precipitation probability (0-1) for a city on a given date."""
    city = city.lower().strip()
    tz_offset = CITIES.get(city, (0, 0, -5))[2]
    if target_date is None:
        now_local = datetime.now(timezone.utc) + timedelta(hours=tz_offset)
        target_date = now_local.date()

    target_str = target_date.strftime("%Y-%m-%d")
    periods = await _fetch_hourly(city)
    probs = []
    for p in periods:
        if _local_date_str(p, tz_offset) != target_str:
            continue
        chance = p.get("probabilityOfPrecipitation", {})
        if isinstance(chance, dict):
            val = chance.get("value")
            if val is not None:
                probs.append(float(val))
    return max(probs) / 100.0 if probs else None
