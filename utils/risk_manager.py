"""
Portfolio-level risk controls.

Hard limits applied before any paper (or live) trade is recorded:
  - Max 10 open positions total
  - Max 10% of starting balance deployed at any time ($100 on $1000)
  - Max 3 positions settling on the same calendar date
  - Circuit breaker: pause new weather trades if recent win rate < 40%
"""
import json
from datetime import datetime
from typing import Optional

MAX_OPEN_POSITIONS  = 10
MAX_DEPLOYED_PCT    = 0.10   # 10% of starting balance
MAX_PER_SETTLE_DATE = 3
CIRCUIT_LOOKBACK    = 20     # trades to look back for win-rate check
CIRCUIT_MIN_WINRATE = 0.40   # halt if win rate drops below 40%


def _settle_date(ticker: str) -> Optional[str]:
    """Extract YYYYMMDD from a Kalshi ticker like KXHIGHNY-26MAY24-T60."""
    import re
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", ticker)
    if not m:
        return None
    try:
        dt = datetime.strptime(f"20{m.group(1)} {m.group(2)} {m.group(3)}", "%Y %b %d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def can_open(platform: str, event: str, cost: float, portfolio: dict) -> tuple[bool, str]:
    """
    Returns (allowed, reason). Call before record_paper_trade.
    portfolio = get_paper_summary() dict.
    """
    starting = portfolio.get("starting_balance", 1000.0)

    # 1. Total deployed cap
    max_deploy = starting * MAX_DEPLOYED_PCT
    if portfolio.get("deployed", 0) + cost > max_deploy:
        return False, f"deployed ${portfolio['deployed']:.0f}+${cost:.0f} > max ${max_deploy:.0f}"

    # 2. Total open positions cap
    if portfolio.get("open_count", 0) >= MAX_OPEN_POSITIONS:
        return False, f"already {portfolio['open_count']} open positions (max {MAX_OPEN_POSITIONS})"

    # 3. Per-settlement-date cap (only for WeatherEdge)
    if platform == "WeatherEdge":
        settle_day = _settle_date(event)
        if settle_day:
            positions = portfolio.get("positions", [])
            open_positions = [p for p in positions if p.get("status") == "open"]
            same_day = sum(
                1 for p in open_positions
                if p.get("platform") == "WeatherEdge" and
                _settle_date(p.get("event", "")) == settle_day
            )
            if same_day >= MAX_PER_SETTLE_DATE:
                return False, f"already {same_day} weather positions settling {settle_day}"

    return True, "ok"


def circuit_open(portfolio: dict) -> bool:
    """
    Returns True if the circuit breaker should halt new weather trades.
    Checks win rate of the last CIRCUIT_LOOKBACK weather settlements.
    """
    positions = portfolio.get("positions", [])
    weather_settled = [
        p for p in positions
        if p.get("status") == "settled" and p.get("platform") == "WeatherEdge"
    ][-CIRCUIT_LOOKBACK:]

    if len(weather_settled) < 5:
        return False  # not enough data to trigger

    wins = sum(1 for p in weather_settled if (p.get("payout") or 0) > (p.get("cost") or 0))
    win_rate = wins / len(weather_settled)
    return win_rate < CIRCUIT_MIN_WINRATE
