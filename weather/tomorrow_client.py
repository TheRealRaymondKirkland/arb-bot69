"""
Tomorrow.io forecast client — better ML forecasts with uncertainty bands.

Free tier: 500 calls/day, sufficient for our use case.
Set TOMORROW_API_KEY in .env to enable.

Falls back gracefully if key not set.
"""
import json
import logging
import os
import time
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

API_KEY    = os.getenv("TOMORROW_API_KEY", "")
BASE_URL   = "https://api.tomorrow.io/v4/weather/forecast"
CACHE_DIR  = Path(__file__).parent.parent / "data"
CACHE_FILE = CACHE_DIR / "tomorrow_forecasts.json"
CACHE_TTL  = 1800   # 30 min

CITIES = {
    "seattle":      (47.6062, -122.3321),
    "austin":       (30.2672,  -97.7431),
    "san antonio":  (29.4241,  -98.4936),
    "houston":      (29.7604,  -95.3698),
    "new york":     (40.7128,  -74.0060),
    "nyc":          (40.7128,  -74.0060),
    "los angeles":  (34.0522, -118.2437),
    "chicago":      (41.8781,  -87.6298),
    "dallas":       (32.7767,  -96.7970),
    "miami":        (25.7617,  -80.1918),
    "phoenix":      (33.4484, -112.0740),
    "denver":       (39.7392, -104.9903),
    "atlanta":      (33.7490,  -84.3880),
}

TZ_OFFSETS = {
    "seattle": -7, "austin": -5, "san antonio": -5, "houston": -5,
    "new york": -4, "nyc": -4, "los angeles": -7, "chicago": -5,
    "dallas": -5, "miami": -4, "phoenix": -7, "denver": -6, "atlanta": -4,
}


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    except Exception:
        return {}


def _save_cache(data: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data, indent=2))


def is_available() -> bool:
    return bool(API_KEY)


async def get_hourly_forecast(city: str) -> list:
    """
    Return list of hourly forecast dicts: {time, temp_f, temp_min_f, temp_max_f, precip_prob}.
    Returns [] if API key not set or request fails.
    """
    if not API_KEY:
        return []

    city = city.lower().strip()
    if city not in CITIES:
        return []

    cache = _load_cache()
    entry = cache.get(city, {})
    if entry and time.time() - entry.get("ts", 0) < CACHE_TTL:
        return entry["hours"]

    lat, lon = CITIES[city]
    params = {
        "location":   f"{lat},{lon}",
        "apikey":     API_KEY,
        "timesteps":  "1h",
        "units":      "imperial",
        "fields":     "temperature,temperatureApparent,precipitationProbability,weatherCode",
    }
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(BASE_URL, params=params, timeout=15)
            r.raise_for_status()
            data  = r.json()
            hours = []
            for interval in data.get("timelines", {}).get("hourly", []):
                vals = interval.get("values", {})
                hours.append({
                    "time":        interval["time"],
                    "temp_f":      vals.get("temperature"),
                    "precip_prob": vals.get("precipitationProbability", 0),
                })
            cache[city] = {"ts": time.time(), "hours": hours}
            _save_cache(cache)
            logger.info(f"Tomorrow.io forecast fetched for {city}: {len(hours)} hours")
            return hours
    except Exception as e:
        logger.warning(f"Tomorrow.io fetch failed for {city}: {e}")
        return []


def _local_hour_date(iso_time: str, tz_offset: int) -> str:
    try:
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        local = dt + timedelta(hours=tz_offset)
        return local.strftime("%Y-%m-%d")
    except Exception:
        return ""


async def get_daily_high(city: str, target_date: Optional[date] = None) -> Optional[float]:
    city = city.lower().strip()
    tz   = TZ_OFFSETS.get(city, -5)
    if target_date is None:
        target_date = (datetime.now(timezone.utc) + timedelta(hours=tz)).date()
    target_str = target_date.strftime("%Y-%m-%d")
    hours = await get_hourly_forecast(city)
    temps = [h["temp_f"] for h in hours if _local_hour_date(h["time"], tz) == target_str and h["temp_f"] is not None]
    return float(max(temps)) if temps else None


async def get_daily_low(city: str, target_date: Optional[date] = None) -> Optional[float]:
    city = city.lower().strip()
    tz   = TZ_OFFSETS.get(city, -5)
    if target_date is None:
        target_date = (datetime.now(timezone.utc) + timedelta(hours=tz)).date()
    target_str = target_date.strftime("%Y-%m-%d")
    hours = await get_hourly_forecast(city)
    temps = [h["temp_f"] for h in hours if _local_hour_date(h["time"], tz) == target_str and h["temp_f"] is not None]
    return float(min(temps)) if temps else None
