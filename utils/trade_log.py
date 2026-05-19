"""
Persistent trade journal. Writes every execution to trades.db (SQLite).
Query daily_pnl() to get today's expected profit + trade count.
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "trades.db"


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT    NOT NULL,
            platform        TEXT    NOT NULL,
            event           TEXT    NOT NULL,
            sets            REAL,
            cost            REAL,
            expected_profit REAL,
            status          TEXT,
            orders          TEXT
        )
    """)
    conn.commit()
    return conn


def log_trade(
    platform: str,
    event: str,
    sets: float,
    cost: float,
    expected_profit: float,
    status: str,
    orders: list,
):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO trades (ts,platform,event,sets,cost,expected_profit,status,orders) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                datetime.now().isoformat(),
                platform,
                event,
                sets,
                cost,
                expected_profit,
                status,
                json.dumps(orders),
            ),
        )


def daily_pnl() -> tuple:
    """Returns (expected_profit_today, trade_count_today) for executed + dry_run trades."""
    today = datetime.now().strftime("%Y-%m-%d")
    with _conn() as conn:
        row = conn.execute(
            "SELECT SUM(expected_profit), COUNT(*) FROM trades "
            "WHERE ts LIKE ? AND status IN ('executed','dry_run')",
            (f"{today}%",),
        ).fetchone()
    return (row[0] or 0.0, row[1] or 0)


def print_daily_summary():
    pnl, count = daily_pnl()
    label = f"${pnl:+.2f}" if pnl != 0 else "$0.00"
    print(f"\n  Today's P&L: {label} from {count} trade(s)")
