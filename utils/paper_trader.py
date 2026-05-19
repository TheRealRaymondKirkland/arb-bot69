"""
Paper trading engine.

Maintains a virtual portfolio in paper_portfolio.db:
  - Starting balance (PAPER_BALANCE env, default $1 000)
  - Open positions — prevents re-buying the same event on every scan
  - Settled positions — marked when a market resolves
  - Running P&L

Key rule: one open position per event ticker at a time.
"""
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "paper_portfolio.db"
STARTING_BALANCE = float(os.getenv("PAPER_BALANCE", "1000.0"))


# ─── Schema ───────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ts               TEXT    NOT NULL,
            platform         TEXT    NOT NULL,
            event            TEXT    NOT NULL,
            sets             REAL,
            cost             REAL    NOT NULL,
            expected_profit  REAL    NOT NULL,
            profit_pct       REAL,
            status           TEXT    NOT NULL DEFAULT 'open',
            settled_ts       TEXT,
            payout           REAL,
            orders           TEXT
        );
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_open_event
            ON portfolio(event) WHERE status='open';
    """)
    # Migrate: drop UNIQUE constraint on event so re-entry is allowed after settlement
    pragma = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='portfolio'"
    ).fetchone()
    if pragma and "UNIQUE" in (pragma["sql"] or "").upper():
        conn.executescript("""
            ALTER TABLE portfolio RENAME TO _portfolio_old;
            CREATE TABLE portfolio (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ts               TEXT    NOT NULL,
                platform         TEXT    NOT NULL,
                event            TEXT    NOT NULL,
                sets             REAL,
                cost             REAL    NOT NULL,
                expected_profit  REAL    NOT NULL,
                profit_pct       REAL,
                status           TEXT    NOT NULL DEFAULT 'open',
                settled_ts       TEXT,
                payout           REAL,
                orders           TEXT
            );
            INSERT INTO portfolio SELECT * FROM _portfolio_old;
            DROP TABLE _portfolio_old;
        """)
    # Seed starting balance on first use
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('starting_balance', ?)",
        (str(STARTING_BALANCE),)
    )
    conn.commit()
    return conn


# ─── Reads ────────────────────────────────────────────────────────────────────

def get_starting_balance() -> float:
    with _conn() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key='starting_balance'").fetchone()
        return float(row["value"]) if row else STARTING_BALANCE


def get_open_positions() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM portfolio WHERE status='open' ORDER BY ts DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_positions(limit: int = 50) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM portfolio ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def is_open(event: str) -> bool:
    with _conn() as conn:
        row = conn.execute(
            "SELECT id FROM portfolio WHERE event=? AND status='open'", (event,)
        ).fetchone()
        return row is not None


def get_paper_summary() -> dict:
    """Return full portfolio state."""
    start = get_starting_balance()
    with _conn() as conn:
        open_rows = conn.execute(
            "SELECT SUM(cost) as total_cost, SUM(expected_profit) as total_ep, COUNT(*) as n "
            "FROM portfolio WHERE status='open'"
        ).fetchone()
        settled_rows = conn.execute(
            """SELECT SUM(payout - cost) as realized, COUNT(*) as n
               FROM portfolio
               WHERE status='settled'"""
        ).fetchone()

    deployed     = open_rows["total_cost"]     or 0.0
    open_ep      = open_rows["total_ep"]       or 0.0
    open_count   = open_rows["n"]              or 0
    realized_pnl = settled_rows["realized"]    or 0.0
    settled_count= settled_rows["n"]           or 0

    virtual_balance = start - deployed + realized_pnl
    total_ep_pnl    = realized_pnl + open_ep

    return {
        "starting_balance": start,
        "virtual_balance":  round(virtual_balance, 4),
        "deployed":         round(deployed, 4),
        "open_ep":          round(open_ep, 4),
        "realized_pnl":     round(realized_pnl, 4),
        "total_ep_pnl":     round(total_ep_pnl, 4),
        "open_count":       open_count,
        "settled_count":    settled_count,
        "positions":        get_all_positions(30),
    }


# ─── Writes ───────────────────────────────────────────────────────────────────

def record_paper_trade(
    platform: str,
    event: str,
    sets: float,
    cost: float,
    expected_profit: float,
    orders: list,
) -> dict:
    """
    Open a new paper position. Returns {"status": "opened"|"duplicate"|"insufficient_funds"}.
    Skips silently if already open for this event.
    """
    if is_open(event):
        return {"status": "duplicate", "event": event}

    summary = get_paper_summary()
    if summary["virtual_balance"] < cost:
        return {"status": "insufficient_funds", "have": summary["virtual_balance"], "need": cost}

    # Profit margin: profit as % of total payout received (matches MECE scanner display)
    # e.g. cost=$0.87, ep=$0.13 → margin=13%, not ROI=14.9%
    payout_total = cost + expected_profit
    profit_pct = (expected_profit / payout_total * 100) if payout_total > 0 else 0
    with _conn() as conn:
        conn.execute(
            """INSERT INTO portfolio
               (ts, platform, event, sets, cost, expected_profit, profit_pct, status, orders)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now().isoformat(),
                platform, event,
                sets, round(cost, 6),
                round(expected_profit, 6),
                round(profit_pct, 2),
                "open",
                json.dumps(orders),
            )
        )
    return {"status": "opened", "event": event, "cost": cost, "expected_profit": expected_profit}


def settle_position(event: str, payout: float) -> bool:
    """Mark an open position as settled with its actual payout."""
    with _conn() as conn:
        n = conn.execute(
            "UPDATE portfolio SET status='settled', settled_ts=?, payout=? "
            "WHERE event=? AND status='open'",
            (datetime.now().isoformat(), payout, event)
        ).rowcount
    return n > 0


def reset_portfolio() -> None:
    """Wipe all positions and reset to starting balance. Use with care."""
    with _conn() as conn:
        conn.execute("DELETE FROM portfolio")
        conn.execute("UPDATE meta SET value=? WHERE key='starting_balance'", (str(STARTING_BALANCE),))
