#!/home/raykirkland88/poly-trading-bot/venv/bin/python
"""
Arb Bot Live Dashboard

Runs the full scan loop and renders a live terminal UI showing:
  - Bot status, balance, uptime
  - Live scan feed (scrolling log)
  - Kalshi MECE opportunities
  - Binary cross-platform matched pairs + arb alerts
  - Today's P&L and trade history

Usage:
  python dashboard.py            # live loop (scans every 60s)
  python dashboard.py --once     # single scan then stay on screen
"""
import asyncio
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

import httpx
from dotenv import load_dotenv
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule

load_dotenv()

from kalshi.kalshi_client import KalshiClient
from cross_platform.binary_scanner import scan_binary_arb
from utils.trade_log import daily_pnl
from utils.paper_trader import (
    get_open_positions, record_paper_trade, settle_position, get_paper_summary,
)
from kalshi.kalshi_client import _get

SCAN_INTERVAL      = int(os.getenv("SCAN_INTERVAL", "60"))
DRY_RUN            = os.getenv("DRY_RUN", "true").lower() != "false"
MIN_PROFIT         = float(os.getenv("MIN_PROFIT", "0.03"))
MAX_POSITION_USDC  = float(os.getenv("MAX_POSITION_USDC", "50"))
SIM_SETTLE_DAYS    = int(os.getenv("SIM_SETTLE_DAYS", "7"))
MAX_LOG_LINES      = 18

console = Console()

# ─── Shared state (updated by scan loop, read by renderer) ───────────────────

state = {
    "balance":        None,
    "last_scan":      None,
    "scan_count":     0,
    "start_time":     time.time(),
    "kalshi_opps":    [],
    "binary_results": None,
    "log":            deque(maxlen=MAX_LOG_LINES),
    "scanning":       False,
    "error":          None,
}


def log(msg: str, style: str = "dim white"):
    ts = datetime.now().strftime("%H:%M:%S")
    state["log"].append((ts, msg, style))


# ─── Layout builders ─────────────────────────────────────────────────────────

def header() -> Panel:
    mode_color = "red" if not DRY_RUN else "yellow"
    mode_label = "LIVE 💰" if not DRY_RUN else "DRY RUN"

    uptime_secs = int(time.time() - state["start_time"])
    h, rem = divmod(uptime_secs, 3600)
    m, s   = divmod(rem, 60)
    uptime = f"{h:02d}:{m:02d}:{s:02d}"

    bal = f"${state['balance']:.2f}" if state["balance"] is not None else "—"
    last = state["last_scan"].strftime("%H:%M:%S") if state["last_scan"] else "—"
    pnl, trades = daily_pnl()
    pnl_color = "green" if pnl >= 0 else "red"
    scan_icon = "[bold yellow]⟳ SCANNING[/]" if state["scanning"] else f"[dim]next in {_next_scan_in()}s[/]"

    title = Text("  ARB BOT — Kalshi × Polymarket  ", style="bold white on dark_blue")

    cols = Table.grid(padding=(0, 3))
    cols.add_row(
        f"[bold cyan]Mode:[/] [{mode_color}]{mode_label}[/]",
        f"[bold cyan]Balance:[/] [white]{bal}[/]",
        f"[bold cyan]Today P&L:[/] [{pnl_color}]${pnl:+.4f}[/] [dim]({trades} trades)[/]",
        f"[bold cyan]Uptime:[/] [white]{uptime}[/]",
        f"[bold cyan]Scans:[/] [white]{state['scan_count']}[/]",
        f"[bold cyan]Last scan:[/] [white]{last}[/]",
        scan_icon,
    )
    return Panel(Align.center(cols), title=title, border_style="dark_blue", padding=(0, 1))


_last_scan_start = [time.time()]

def _next_scan_in() -> int:
    elapsed = time.time() - _last_scan_start[0]
    return max(0, int(SCAN_INTERVAL - elapsed))


def kalshi_panel() -> Panel:
    opps = state["kalshi_opps"]
    t = Table(box=box.SIMPLE_HEAD, show_header=True, expand=True, padding=(0, 1))
    t.add_column("Event", style="white", ratio=4)
    t.add_column("Profit", justify="right", style="bold green", ratio=1)
    t.add_column("Max $", justify="right", style="cyan", ratio=1)
    t.add_column("Safe?", justify="center", ratio=1)
    t.add_column("Volume", justify="right", style="dim", ratio=1)

    if not opps:
        t.add_row("[dim]No MECE arb opportunities right now[/]", "", "", "", "")
    else:
        for o in opps[:8]:
            safe = "[green]✓[/]" if o["has_catchall"] else "[yellow]⚠[/]"
            max_earn = o["profit"] * o["min_size"]
            t.add_row(
                o["title"][:55],
                f"{o['profit']*100:.1f}%",
                f"${max_earn:.2f}",
                safe,
                f"${o['total_volume']:,.0f}",
            )

    title = f"[bold]Kalshi MECE Arb[/]  [dim]({len(opps)} opps)[/]"
    return Panel(t, title=title, border_style="blue", padding=(0, 0))


def binary_panel() -> Panel:
    res = state["binary_results"]
    if res is None:
        return Panel("[dim]Waiting for first scan…[/]", title="[bold]Binary Cross-Platform Arb[/]", border_style="magenta")

    opps    = res["opportunities"]
    pairs   = res["candidates"]
    k_count = res["kalshi_count"]
    p_count = res["poly_count"]

    t = Table(box=box.SIMPLE_HEAD, show_header=True, expand=True, padding=(0, 1))
    t.add_column("Match", justify="center", style="cyan", width=6)
    t.add_column("Kalshi market", style="white", ratio=3)
    t.add_column("Polymarket question", style="white", ratio=3)
    t.add_column("K YES", justify="right", width=7)
    t.add_column("P YES", justify="right", width=7)
    t.add_column("Gap", justify="right", width=8)
    t.add_column("Action", width=14)

    displayed = 0
    # Show confirmed arbs first
    for arb in opps:
        side = ("K✓+PNO" if arb.direction == "buy_kalshi_yes_poly_no"
                else "P✓+KNO")
        t.add_row(
            f"[bold green]{arb.score:.2f}[/]",
            arb.kalshi_title[:42],
            arb.poly_question[:42],
            f"${arb.k_yes_ask:.3f}",
            f"${arb.p_yes_ask:.3f}",
            f"[bold green]+{arb.profit_pct*100:.1f}%[/]",
            f"[bold green]{side}[/]",
        )
        displayed += 1

    # Fill remaining rows with top candidates
    for km, pm, score in pairs:
        if displayed >= 10:
            break
        gap1 = 1.0 - km["yes_ask"] - pm["no_snap"]
        gap2 = 1.0 - pm["yes_snap"] - km["no_ask"]
        best = max(gap1, gap2)
        direction = "K→+P←" if gap1 >= gap2 else "P→+K←"
        gap_str = (f"[bold green]+{best*100:.1f}%[/]" if best >= MIN_PROFIT
                   else f"[dim]{best*100:+.1f}%[/]")
        t.add_row(
            f"[white]{score:.2f}[/]",
            km["title"][:42],
            pm["question"][:42],
            f"${km['yes_ask']:.3f}",
            f"${pm['yes_snap']:.3f}",
            gap_str,
            f"[dim]{direction}[/]" if best < MIN_PROFIT else f"[yellow]{direction}[/]",
        )
        displayed += 1

    arb_badge = (f"[bold green] {len(opps)} ARB{'S' if len(opps)!=1 else ''} LIVE [/]"
                 if opps else "[dim] no arb yet [/]")
    title = (f"[bold]Binary Cross-Platform[/]  {arb_badge}  "
             f"[dim]Kalshi pool {k_count} · Poly {p_count} · {len(pairs)} pairs[/]")
    border = "green" if opps else "magenta"
    return Panel(t, title=title, border_style=border, padding=(0, 0))


def log_panel() -> Panel:
    lines = list(state["log"])
    t = Table.grid(padding=(0, 1), expand=True)
    t.add_column("Time", style="dim", width=10)
    t.add_column("Message", ratio=1)

    if not lines:
        t.add_row("", "[dim]Waiting for scan output…[/]")
    for ts, msg, style in lines:
        t.add_row(f"[dim]{ts}[/]", f"[{style}]{msg}[/]")

    return Panel(t, title="[bold]Scan Log[/]", border_style="grey50", padding=(0, 0))


def build_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="main"),
        Layout(name="log", size=MAX_LOG_LINES + 3),
    )
    layout["main"].split_row(
        Layout(name="kalshi", ratio=2),
        Layout(name="binary", ratio=3),
    )
    layout["header"].update(header())
    layout["kalshi"].update(kalshi_panel())
    layout["binary"].update(binary_panel())
    layout["log"].update(log_panel())
    return layout


# ─── Scan loop ────────────────────────────────────────────────────────────────

async def settle_open_positions(client: httpx.AsyncClient):
    positions = [p for p in get_open_positions() if p["platform"] in ("Kalshi", "BinaryArb")]
    if not positions:
        return
    sem = asyncio.Semaphore(5)

    async def check_one(pos):
        async with sem:
            event = pos["event"]
            sets  = float(pos["sets"] or 1.0)
            cost  = float(pos["cost"])
            try:
                data   = await _get(client, "/markets", {"event_ticker": event, "limit": 50})
                winner = next((m for m in data.get("markets", [])
                               if m.get("result") == "yes"), None)
                if winner:
                    payout = round(sets * 1.0, 4)
                    if settle_position(event, payout):
                        log(f"[PAPER] settled (resolved): {event}  pnl=${payout-cost:+.4f}", "bold green")
                    return
            except Exception:
                pass
            if SIM_SETTLE_DAYS > 0:
                age = (datetime.now() - datetime.fromisoformat(pos["ts"])).days
                if age >= SIM_SETTLE_DAYS:
                    payout = round(cost + float(pos["expected_profit"]), 4)
                    if settle_position(event, payout):
                        log(f"[PAPER] settled (sim {age}d): {event}  pnl=+${pos['expected_profit']:.4f}", "green")

    await asyncio.gather(*[check_one(p) for p in positions])


async def run_scan(client: httpx.AsyncClient, kalshi: KalshiClient):
    state["scanning"] = True
    _last_scan_start[0] = time.time()
    log("── scan started ──", "dim cyan")

    try:
        # Settle resolved paper positions before scanning
        await settle_open_positions(client)

        # Balance
        try:
            bal = await kalshi.get_balance(client)
            state["balance"] = bal
            log(f"Balance: ${bal:.2f}", "cyan")
        except Exception as e:
            log(f"Balance fetch failed: {e}", "red")

        # Kalshi MECE
        log("Scanning Kalshi MECE markets…", "dim white")
        opps = []
        try:
            opps = await kalshi.scan_mece_opportunities(client, min_profit=float(os.getenv("MIN_PROFIT", "0.04")))
            state["kalshi_opps"] = opps
            log(f"Kalshi MECE: {len(opps)} arb opportunities", "white" if opps else "dim white")
            for o in opps[:3]:
                safe = "SAFE" if o["has_catchall"] else "INCOMPLETE"
                log(f"  {o['event_ticker']} {o['profit']*100:.1f}% profit [{safe}]",
                    "green" if o["has_catchall"] else "yellow")
            # Paper trade MECE arbs (capped at MAX_POSITION_USDC)
            if DRY_RUN:
                for o in opps:
                    if o["min_size"] < 0.01:
                        continue
                    sets = round(min(o["min_size"], MAX_POSITION_USDC / o["sum_yes_ask"]), 4)
                    cost   = round(o["sum_yes_ask"] * sets, 6)
                    profit = round(o["profit"] * sets, 6)
                    pt = record_paper_trade(
                        platform="Kalshi", event=o["event_ticker"], sets=sets,
                        cost=cost, expected_profit=profit,
                        orders=[{"ticker": m.get("ticker"), "at_price": float(m.get("yes_ask_dollars", 0))}
                                for m in o.get("markets", [])],
                    )
                    if pt["status"] == "opened":
                        flag = "SAFE" if o["has_catchall"] else "INCOMPLETE"
                        log(f"  [PAPER] opened [{flag}]: {o['event_ticker']}  cost=${cost:.4f}  ep=${profit:.4f}", "bold green")
                    elif pt["status"] == "duplicate":
                        log(f"  [PAPER] holding: {o['event_ticker']}", "dim white")
                    elif pt["status"] == "insufficient_funds":
                        log(f"  [PAPER] low balance, skip: {o['event_ticker']}", "yellow")
        except Exception as e:
            log(f"MECE scan error: {e}", "red")

        # Binary cross-platform — reuse MECE opps to avoid double scan
        log("Scanning binary cross-platform arb…", "dim white")
        try:
            res = await scan_binary_arb(client, min_profit=MIN_PROFIT, mece_opps=opps,
                                        max_position_usdc=MAX_POSITION_USDC)
            state["binary_results"] = res
            n_pairs = len(res["candidates"])
            n_arbs  = len(res["opportunities"])
            log(f"Binary scan: {n_pairs} matched pairs, {res['kalshi_count']} Kalshi, {res['poly_count']} Poly", "white")
            if n_arbs:
                for arb in res["opportunities"]:
                    log(f"  *** ARB: {arb.profit_pct*100:.2f}% — {arb.kalshi_title[:40]}", "bold green")
                    if DRY_RUN and arb.max_contracts > 0:
                        cost   = round((arb.kalshi_leg + arb.poly_leg) * arb.max_contracts, 6)
                        profit = round(arb.profit_pct * arb.max_contracts, 6)
                        k_side = "yes" if arb.direction == "buy_kalshi_yes_poly_no" else "no"
                        p_side = "no"  if arb.direction == "buy_kalshi_yes_poly_no" else "yes"
                        pt = record_paper_trade(
                            platform="BinaryArb", event=arb.kalshi_ticker,
                            sets=arb.max_contracts, cost=cost, expected_profit=profit,
                            orders=[
                                {"ticker": arb.kalshi_ticker, "side": f"kalshi_{k_side}", "at_price": arb.kalshi_leg},
                                {"ticker": arb.poly_market_id, "side": f"poly_{p_side}", "at_price": arb.poly_leg},
                            ],
                        )
                        if pt["status"] == "opened":
                            log(f"  [PAPER] binary arb opened: {arb.kalshi_ticker}  cost=${cost:.4f}  ep=${profit:.4f}", "bold green")
                        elif pt["status"] == "duplicate":
                            log(f"  [PAPER] holding binary: {arb.kalshi_ticker}", "dim white")
            else:
                # Show best gap
                if res["candidates"]:
                    km, pm, score = res["candidates"][0]
                    gap = max(1 - km["yes_ask"] - pm["no_snap"],
                              1 - pm["yes_snap"] - km["no_ask"])
                    log(f"  Best gap: {gap*100:+.1f}% [{score:.2f}] {km['title'][:35]}…", "dim white")
        except Exception as e:
            log(f"Binary scan error: {e}", "red")

    except Exception as e:
        log(f"Scan crashed: {e}", "bold red")
        state["error"] = str(e)
    finally:
        state["scanning"] = False
        state["scan_count"] += 1
        state["last_scan"] = datetime.now()
        elapsed = time.time() - _last_scan_start[0]
        log(f"── scan done in {elapsed:.1f}s ──", "dim cyan")


async def scan_loop(live: Live):
    async with httpx.AsyncClient() as client:
        kalshi = KalshiClient()
        run_once = "--once" in sys.argv

        while True:
            await run_scan(client, kalshi)
            live.update(build_layout())

            if run_once:
                log("─ [dim]--once flag set, staying on screen[/] ─", "dim")
                # Keep refreshing display but don't scan again
                while True:
                    await asyncio.sleep(1)
                    live.update(build_layout())

            await asyncio.sleep(1)  # tick to refresh display
            deadline = _last_scan_start[0] + SCAN_INTERVAL
            while time.time() < deadline:
                await asyncio.sleep(1)
                live.update(build_layout())


# ─── Entry point ─────────────────────────────────────────────────────────────

async def main():
    log("Dashboard starting…", "dim white")
    layout = build_layout()

    with Live(layout, console=console, refresh_per_second=2, screen=True) as live:
        await scan_loop(live)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard stopped.[/]")
