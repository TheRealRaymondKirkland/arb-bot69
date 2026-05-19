"""
Open-Meteo forecast client — free, no API key required.
https://github.com/open-meteo/open-meteo

Uses the ICON Seamless ensemble model (40 members) to compute
real probability distributions for temperature buckets.

Instead of assuming sigma=3°F Gaussian error, we ask 40 independent
weather model runs what they predict and count votes:
  P(high in 68–69°F) = (members with daily high in that range) / 40

This is how professional forecasters actually compute probabilities.
"""
import asyncio
import json
import logging
import time
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
POINT_URL    = "https://api.open-meteo.com/v1/forecast"
CACHE_DIR    = Path(__file__).parent.parent / "data"
CACHE_FILE   = CACHE_DIR / "openmeteo_cache.json"
CACHE_TTL    = 3600   # 1 hour (ensemble models update every 6h)

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


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    except Exception:
        return {}


def _save_cache(data: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data, indent=2))


async def _fetch_ensemble(city: str) -> dict:
    """
    Fetch hourly ensemble members for a city.
    Returns {member_key: [temp, temp, ...], "time": [...]} or {}.
    Cached for CACHE_TTL seconds.
    """
    city = city.lower().strip()
    if city not in CITIES:
        return {}

    cache = _load_cache()
    entry = cache.get(city, {})
    if entry and time.time() - entry.get("ts", 0) < CACHE_TTL:
        return entry.get("hourly", {})

    lat, lon, _ = CITIES[city]
    params = {
        "latitude":           lat,
        "longitude":          lon,
        "hourly":             "temperature_2m",
        "models":             "icon_seamless",
        "temperature_unit":   "fahrenheit",
        "timezone":           "UTC",
        "forecast_days":      7,
    }
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(ENSEMBLE_URL, params=params, timeout=20)
            r.raise_for_status()
            data   = r.json()
            hourly = data.get("hourly", {})
            members = [k for k in hourly if k.startswith("temperature_2m_member")]
            if not members:
                logger.warning(f"Open-Meteo ensemble returned no members for {city}")
                return {}
            cache[city] = {"ts": time.time(), "hourly": hourly}
            _save_cache(cache)
            logger.info(
                f"Open-Meteo ensemble fetched for {city}: "
                f"{len(hourly.get('time', []))} hours × {len(members)} members"
            )
            return hourly
    except Exception as e:
        logger.warning(f"Open-Meteo ensemble fetch failed for {city}: {e}")
        return {}


async def _fetch_point(city: str) -> list:
    """
    Fallback: fetch a single point forecast (mean model) when ensemble unavailable.
    Returns list of {time, temp_f} dicts.
    """
    city = city.lower().strip()
    if city not in CITIES:
        return []

    lat, lon, _ = CITIES[city]
    params = {
        "latitude":         lat,
        "longitude":        lon,
        "hourly":           "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone":         "UTC",
        "forecast_days":    7,
    }
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(POINT_URL, params=params, timeout=15)
            r.raise_for_status()
            data  = r.json()
            times = data.get("hourly", {}).get("time", [])
            temps = data.get("hourly", {}).get("temperature_2m", [])
            return [{"time": t, "temp_f": v} for t, v in zip(times, temps) if v is not None]
    except Exception as e:
        logger.warning(f"Open-Meteo point fetch failed for {city}: {e}")
        return []


def _date_hours(hourly: dict, tz_offset: int, target_date: date) -> list[int]:
    """Return indices into hourly['time'] for the given local date."""
    target_str = target_date.strftime("%Y-%m-%d")
    indices = []
    for i, t_str in enumerate(hourly.get("time", [])):
        try:
            dt    = datetime.fromisoformat(t_str)
            local = dt + timedelta(hours=tz_offset)
            if local.strftime("%Y-%m-%d") == target_str:
                indices.append(i)
        except Exception:
            pass
    return indices


def bucket_prob(members: list[float], low_f: Optional[float], high_f: Optional[float]) -> float:
    """
    Fraction of ensemble members whose temperature falls in [low_f, high_f].
    Uses half-degree bucket boundaries (matching Kalshi's settlement rules).
    """
    if not members:
        return 0.5
    lo = (low_f  - 0.5) if low_f  is not None else -1e9
    hi = (high_f + 0.5) if high_f is not None else  1e9
    return sum(1 for m in members if lo <= m <= hi) / len(members)


async def get_member_extremes(
    city: str,
    target_date: Optional[date] = None,
    metric: str = "high",
) -> Optional[list[float]]:
    """
    Return list of per-member daily high (or low) temps for a city on a date.
    e.g. [82.1, 84.3, 81.7, ...] — one value per ensemble member.
    Returns None if data unavailable.
    """
    city = city.lower().strip()
    tz   = CITIES.get(city, (0, 0, -5))[2]
    if target_date is None:
        target_date = (datetime.now(timezone.utc) + timedelta(hours=tz)).date()

    hourly  = await _fetch_ensemble(city)
    if not hourly:
        return None

    indices = _date_hours(hourly, tz, target_date)
    if not indices:
        return None

    member_keys = sorted(k for k in hourly if k.startswith("temperature_2m_member"))
    if not member_keys:
        return None

    extremes = []
    for k in member_keys:
        series = hourly[k]
        vals   = [series[i] for i in indices if i < len(series) and series[i] is not None]
        if not vals:
            continue
        extremes.append(max(vals) if metric == "high" else min(vals))

    return extremes if extremes else None


async def get_bucket_prob(
    city: str,
    target_date: date,
    metric: str,
    low_f: Optional[float],
    high_f: Optional[float],
) -> Optional[float]:
    """
    Compute P(daily metric falls in [low_f, high_f]) from ensemble.
    Returns None if ensemble data unavailable (caller should fall back to Gaussian).
    """
    members = await get_member_extremes(city, target_date, metric)
    if not members:
        return None
    return bucket_prob(members, low_f, high_f)


async def get_daily_high(city: str, target_date: Optional[date] = None) -> Optional[float]:
    """Return ensemble mean daily high temp (°F) — for display/logging only."""
    members = await get_member_extremes(city, target_date, "high")
    if members:
        return round(sum(members) / len(members), 1)
    # Fallback to point forecast
    city = city.lower().strip()
    tz   = CITIES.get(city, (0, 0, -5))[2]
    if target_date is None:
        target_date = (datetime.now(timezone.utc) + timedelta(hours=tz)).date()
    hours = await _fetch_point(city)
    target_str = target_date.strftime("%Y-%m-%d")
    temps = [
        h["temp_f"] for h in hours
        if h["time"][:10] == target_str and h["temp_f"] is not None
    ]
    return float(max(temps)) if temps else None


async def get_daily_low(city: str, target_date: Optional[date] = None) -> Optional[float]:
    """Return ensemble mean daily low temp (°F) — for display/logging only."""
    members = await get_member_extremes(city, target_date, "low")
    if members:
        return round(sum(members) / len(members), 1)
    city = city.lower().strip()
    tz   = CITIES.get(city, (0, 0, -5))[2]
    if target_date is None:
        target_date = (datetime.now(timezone.utc) + timedelta(hours=tz)).date()
    hours = await _fetch_point(city)
    target_str = target_date.strftime("%Y-%m-%d")
    temps = [
        h["temp_f"] for h in hours
        if h["time"][:10] == target_str and h["temp_f"] is not None
    ]
    return float(min(temps)) if temps else None


async def prefetch_all(cities: Optional[list] = None) -> None:
    """Warm ensemble cache for all cities in parallel."""
    targets = cities or list(CITIES.keys())
    await asyncio.gather(*[_fetch_ensemble(c) for c in targets], return_exceptions=True)


def is_available() -> bool:
    return True   # always available, no API key needed
