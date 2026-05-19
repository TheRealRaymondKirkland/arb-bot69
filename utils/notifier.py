"""
ntfy push notifications for paper trading events.

Three channels (subscribe in the ntfy app):
  {PREFIX}-open   — new position opened
  {PREFIX}-close  — position settled / closed
  {PREFIX}-pnl    — realized P&L changed

Set NTFY_TOPIC_PREFIX in .env (e.g. "rayjkarbbot").
Topics are public on ntfy.sh — pick something unique.
"""
import asyncio
import logging
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_PREFIX = os.getenv("NTFY_TOPIC_PREFIX", "arb-bot")
NTFY_BASE = "https://ntfy.sh"

TOPIC_OPEN  = f"{_PREFIX}-open"
TOPIC_CLOSE = f"{_PREFIX}-close"
TOPIC_PNL   = f"{_PREFIX}-pnl"


async def _send(topic: str, message: str, title: str, tags: str, priority: int = 3) -> None:
    headers = {
        "Title":    title,
        "Tags":     tags,
        "Priority": str(priority),
    }
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{NTFY_BASE}/{topic}",
                content=message.encode(),
                headers=headers,
                timeout=6,
            )
            r.raise_for_status()
    except Exception as e:
        logger.warning(f"ntfy failed [{topic}]: {e}")


def fire(coro) -> None:
    """Schedule a notification coroutine without blocking the caller."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(coro)
        else:
            loop.run_until_complete(coro)
    except Exception as e:
        logger.warning(f"ntfy fire error: {e}")


# ─── Public helpers ───────────────────────────────────────────────────────────

def notify_open(
    event: str,
    platform: str,
    sets: float,
    cost: float,
    expected_profit: float,
) -> None:
    roi = expected_profit / cost * 100 if cost > 0 else 0
    msg = (
        f"{platform} | {event}\n"
        f"Sets: {sets}  Cost: ${cost:.2f}  Expected: +${expected_profit:.2f} ({roi:.1f}% ROI)"
    )
    fire(_send(TOPIC_OPEN, msg, title="Position Opened", tags="chart_increasing", priority=3))


def notify_close(
    event: str,
    platform: str,
    cost: float,
    payout: float,
) -> None:
    pnl  = payout - cost
    won  = pnl >= 0
    sign = "+" if won else ""
    msg  = (
        f"{platform} | {event}\n"
        f"Cost: ${cost:.2f}  Payout: ${payout:.2f}  P&L: {sign}${pnl:.2f}"
    )
    tag  = "white_check_mark" if won else "x"
    pri  = 4 if won else 3
    fire(_send(TOPIC_CLOSE, msg, title="Position Closed", tags=tag, priority=pri))


def notify_pnl(
    realized_pnl: float,
    delta: float,
    settled_count: int,
) -> None:
    sign = "+" if delta >= 0 else ""
    msg  = (
        f"Realized P&L: ${realized_pnl:.2f}  ({sign}${delta:.2f} this cycle)\n"
        f"{settled_count} total settled trades"
    )
    tag = "moneybag" if delta >= 0 else "chart_with_downwards_trend"
    fire(_send(TOPIC_PNL, msg, title="P&L Update", tags=tag, priority=2))
