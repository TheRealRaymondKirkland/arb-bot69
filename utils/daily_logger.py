"""
Transaction log and daily summary generator.

Transaction log  →  logs/trades_YYYY-MM-DD.csv   (one row per open/close)
Daily summary    →  logs/summary_YYYY-MM-DD.txt  (human-readable report)

Both are keyed to the current calendar day so they rotate automatically.
"""
import csv
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"
DB_PATH  = Path(__file__).parent.parent / "paper_portfolio.db"


def _logs_dir() -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    return LOGS_DIR


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ─── Transaction log (CSV) ────────────────────────────────────────────────────

def _tx_path(date: str = None) -> Path:
    return _logs_dir() / f"trades_{date or _today()}.csv"


_CSV_FIELDS = [
    "ts", "type", "platform", "event", "sets",
    "cost", "expected_profit", "payout", "pnl", "roi_pct",
]


def _ensure_header(path: Path) -> None:
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()


def log_open(
    platform: str,
    event: str,
    sets: float,
    cost: float,
    expected_profit: float,
) -> None:
    path = _tx_path()
    _ensure_header(path)
    roi = expected_profit / cost * 100 if cost > 0 else 0
    row = {
        "ts":              datetime.now().isoformat(timespec="seconds"),
        "type":            "OPEN",
        "platform":        platform,
        "event":           event,
        "sets":            round(sets, 4),
        "cost":            round(cost, 4),
        "expected_profit": round(expected_profit, 4),
        "payout":          "",
        "pnl":             "",
        "roi_pct":         round(roi, 2),
    }
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(row)


def log_close(
    platform: str,
    event: str,
    sets: float,
    cost: float,
    payout: float,
) -> None:
    # Find the open record to pull expected_profit / sets if not passed
    path = _tx_path()
    _ensure_header(path)
    pnl = payout - cost
    roi = pnl / cost * 100 if cost > 0 else 0
    row = {
        "ts":              datetime.now().isoformat(timespec="seconds"),
        "type":            "CLOSE",
        "platform":        platform,
        "event":           event,
        "sets":            round(sets, 4),
        "cost":            round(cost, 4),
        "expected_profit": "",
        "payout":          round(payout, 4),
        "pnl":             round(pnl, 4),
        "roi_pct":         round(roi, 2),
    }
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(row)


# ─── Summary generator ────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def generate_summary(date: str = None) -> str:
    """
    Build a full text summary for a given date (default: today).
    Queries paper_portfolio.db directly.
    Returns the report as a string and also saves it to logs/.
    """
    date = date or _today()
    date_prefix = date + "T"

    with _conn() as conn:
        # Positions opened on this date
        opened = conn.execute(
            "SELECT * FROM portfolio WHERE ts LIKE ? ORDER BY ts",
            (date_prefix + "%",)
        ).fetchall()

        # Positions settled on this date
        closed = conn.execute(
            "SELECT * FROM portfolio WHERE settled_ts LIKE ? ORDER BY settled_ts",
            (date_prefix + "%",)
        ).fetchall()

        # All currently open positions
        still_open = conn.execute(
            "SELECT * FROM portfolio WHERE status='open' ORDER BY ts DESC"
        ).fetchall()

        # Portfolio meta
        meta = conn.execute("SELECT value FROM meta WHERE key='starting_balance'").fetchone()
        starting = float(meta["value"]) if meta else 1000.0

        # Aggregate realized P&L (all time, not just today)
        agg = conn.execute(
            "SELECT SUM(payout - cost) as realized, COUNT(*) as n "
            "FROM portfolio WHERE status='settled'"
        ).fetchone()
        total_realized = agg["realized"] or 0.0
        total_settled  = agg["n"] or 0

        # Open positions aggregate
        open_agg = conn.execute(
            "SELECT SUM(cost) as deployed, SUM(expected_profit) as ep FROM portfolio WHERE status='open'"
        ).fetchone()
        deployed = open_agg["deployed"] or 0.0
        open_ep  = open_agg["ep"] or 0.0

    virtual_balance = starting + total_realized - deployed
    total_return_pct = (total_realized / starting * 100) if starting > 0 else 0

    lines = []
    w = 58

    def rule(ch="─"):
        lines.append(ch * w)

    def section(title):
        lines.append("")
        lines.append(f"  {title}")
        rule()

    rule("═")
    lines.append(f"  ARB BOT DAILY SUMMARY  ·  {date}")
    rule("═")

    # ── Portfolio snapshot ──
    section("PORTFOLIO SNAPSHOT")
    lines.append(f"  Starting balance   : ${starting:>10,.2f}")
    lines.append(f"  Realized P&L       : ${total_realized:>+10.2f}   ({total_settled} settled trades)")
    lines.append(f"  Deployed (open)    : ${deployed:>10,.2f}   ({len(still_open)} positions)")
    lines.append(f"  Unrealized (EP)    : ${open_ep:>+10.2f}")
    lines.append(f"  Virtual balance    : ${virtual_balance:>10,.2f}")
    lines.append(f"  Total return       : {total_return_pct:>+9.2f}%")

    # ── Opened today ──
    section(f"OPENED TODAY  ({len(opened)})")
    if opened:
        lines.append(f"  {'Event':<32}  {'Sets':>5}  {'Cost':>8}  {'EP':>8}  {'ROI':>6}")
        rule()
        for p in opened:
            ep  = float(p["expected_profit"])
            cst = float(p["cost"])
            roi = ep / cst * 100 if cst > 0 else 0
            lines.append(
                f"  {p['event']:<32}  {float(p['sets'] or 0):>5.0f}"
                f"  ${cst:>7.2f}  ${ep:>+7.2f}  {roi:>5.1f}%"
            )
    else:
        lines.append("  (none)")

    # ── Closed today ──
    section(f"CLOSED TODAY  ({len(closed)})")
    day_realized = sum(float(p["payout"]) - float(p["cost"]) for p in closed)
    if closed:
        lines.append(f"  {'Event':<32}  {'Cost':>8}  {'Payout':>8}  {'P&L':>8}  {''}")
        rule()
        for p in closed:
            cst    = float(p["cost"])
            payout = float(p["payout"])
            pnl    = payout - cst
            flag   = "✓" if pnl >= 0 else "✗"
            lines.append(
                f"  {p['event']:<32}  ${cst:>7.2f}  ${payout:>7.2f}  ${pnl:>+7.2f}  {flag}"
            )
        lines.append("")
        lines.append(f"  Day realized P&L   : ${day_realized:>+.2f}")
    else:
        lines.append("  (none)")

    # ── Still open ──
    section(f"OPEN POSITIONS  ({len(still_open)})")
    if still_open:
        lines.append(f"  {'Event':<32}  {'Sets':>5}  {'Cost':>8}  {'EP':>8}  {'Age':>6}")
        rule()
        now = datetime.now()
        for p in still_open:
            try:
                age_h = (now - datetime.fromisoformat(p["ts"])).total_seconds() / 3600
                age_s = f"{age_h:.1f}h"
            except Exception:
                age_s = "?"
            ep  = float(p["expected_profit"])
            cst = float(p["cost"])
            lines.append(
                f"  {p['event']:<32}  {float(p['sets'] or 0):>5.0f}"
                f"  ${cst:>7.2f}  ${ep:>+7.2f}  {age_s:>6}"
            )
    else:
        lines.append("  (none)")

    # ── Transaction log file reference ──
    tx_file = _tx_path(date)
    lines.append("")
    rule("═")
    lines.append(f"  Transaction log : {tx_file}")
    lines.append(f"  Generated       : {datetime.now().isoformat(timespec='seconds')}")
    rule("═")

    report = "\n".join(lines)

    # Save to file
    summary_path = _logs_dir() / f"summary_{date}.txt"
    summary_path.write_text(report)

    return report


def get_tx_log_path(date: str = None) -> Path:
    return _tx_path(date)
