"""
Dynamic MECE event discovery for Kalshi.

Paginates through open Kalshi events, collecting MECE events directly.
Events are cached to avoid redundant API calls on every scan.

API call budget:
  Discovery (cache miss): DISCOVERY_PAGES calls (5 × 200 = 1 000 events)
  Discovery (cache hit):  0 calls — returns cached events directly
  Per-scan market checks: len(events) calls via scan_mece_opportunities
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

from kalshi.kalshi_client import _get

logger = logging.getLogger(__name__)

CACHE_PATH       = Path(__file__).parent.parent / "data" / "mece_events_cache.json"
CACHE_TTL_HOURS  = 0.5   # 30 minutes — fast enough to catch new events
DISCOVERY_PAGES  = 5      # 5 × 200 = 1 000 events — wide enough for all known MECE
EVENTS_PER_PAGE  = 200


def _load_cache() -> Optional[list[dict]]:
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text())
        age_hours = (time.time() - data["ts"]) / 3600
        if age_hours > CACHE_TTL_HOURS:
            return None
        return data["events"]
    except json.JSONDecodeError as e:
        logger.warning(f"Corrupted MECE cache — will rebuild: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to load MECE cache: {e}")
        return None


def _save_cache(events: list[dict]) -> None:
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps({"ts": time.time(), "events": events}, indent=2))


async def get_mece_events(client: httpx.AsyncClient) -> list[dict]:
    """
    Return all open, mutually-exclusive Kalshi events.

    On cache hit:  0 API calls.
    On cache miss: DISCOVERY_PAGES calls (collects events during pagination,
                   no per-series re-fetch needed).
    """
    cached = _load_cache()
    if cached is not None:
        logger.info(f"MECE event cache hit: {len(cached)} events")
        return cached

    logger.info("Discovering MECE events via Kalshi pagination...")
    events: list[dict] = []
    cursor: Optional[str] = None

    for page in range(DISCOVERY_PAGES):
        params: dict = {"status": "open", "limit": EVENTS_PER_PAGE}
        if cursor:
            params["cursor"] = cursor
        try:
            data = await _get(client, "/events", params)
        except Exception as e:
            logger.warning(f"Discovery page {page + 1} failed: {e}")
            break

        page_events = data.get("events", [])
        mece = [e for e in page_events if e.get("mutually_exclusive")]
        events.extend(mece)
        cursor = data.get("cursor")

        logger.info(
            f"Page {page + 1}: {len(page_events)} events, {len(mece)} MECE "
            f"(total so far: {len(events)}) | more={'yes' if cursor else 'no'}"
        )

        if not cursor or not page_events:
            break

    logger.info(f"Discovered {len(events)} MECE events — caching for {CACHE_TTL_HOURS}h")
    _save_cache(events)
    return events
