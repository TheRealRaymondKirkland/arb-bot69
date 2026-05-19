#!/home/raykirkland88/poly-trading-bot/venv/bin/python
"""
Arb Bot Web Dashboard — FastAPI + WebSocket backend.

Serves a hacker-terminal-style browser UI at http://localhost:8080
Pushes live scan data to all connected clients via WebSocket.

Usage:
  python web_dashboard.py
"""
import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from kalshi.kalshi_client import KalshiClient, _get
from cross_platform.binary_scanner import scan_binary_arb
from utils.trade_log import daily_pnl
from utils.paper_trader import get_paper_summary, record_paper_trade, get_open_positions, settle_position
from utils.notifier import notify_open, notify_close, notify_pnl
from utils.daily_logger import log_open, log_close, generate_summary

logging.basicConfig(level=logging.INFO)

SCAN_INTERVAL      = int(os.getenv("SCAN_INTERVAL", "60"))
DRY_RUN            = os.getenv("DRY_RUN", "true").lower() != "false"
MIN_PROFIT         = float(os.getenv("MIN_PROFIT", "0.03"))
MAX_PROFIT         = float(os.getenv("MAX_PROFIT", "0.15"))
MAX_DAILY_LOSS     = float(os.getenv("MAX_DAILY_LOSS", "20"))
PORT               = int(os.getenv("DASHBOARD_PORT", "8080"))
MAX_POSITION_USDC  = float(os.getenv("MAX_POSITION_USDC", "50"))
SIM_SETTLE_HOURS   = float(os.getenv("SIM_SETTLE_HOURS", "24"))
SETTLE_CHECK_INTERVAL = 900   # Poll Kalshi API for real resolution every 15 min

# ─── Shared state ─────────────────────────────────────────────────────────────

state = {
    "mode":           "DRY RUN" if DRY_RUN else "LIVE",
    "balance":        None,
    "scan_count":     0,
    "start_time":     time.time(),
    "last_scan_ts":   None,
    "last_scan_dur":  None,
    "scanning":       False,
    "kalshi_opps":    [],
    "binary":         None,
    "log":            deque(maxlen=80),
    "arb_alerts":     deque(maxlen=20),
    "next_scan_in":   SCAN_INTERVAL,
    "paper":          None,
    "_current_day":   datetime.now().strftime("%Y-%m-%d"),
}

clients: set[WebSocket] = set()


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str, level: str = "info"):
    entry = {"ts": ts(), "msg": msg, "level": level}
    state["log"].append(entry)


async def broadcast(event: str, data: dict):
    dead = set()
    payload = json.dumps({"event": event, "data": data})
    for ws in list(clients):  # snapshot so mid-await disconnects don't cause RuntimeError
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


# ─── Scan loop ────────────────────────────────────────────────────────────────

async def settle_open_positions(client: httpx.AsyncClient):
    """
    Two-pass settlement:
      1. Real: ask Kalshi API if market resolved (result='yes') — throttled to once/hour.
      2. Simulated: auto-close positions older than SIM_SETTLE_HOURS at expected payout.
    MECE arb payout = sets × $1 (exactly one leg always wins $1/contract).
    """
    positions = [p for p in get_open_positions() if p["platform"] == "Kalshi"]
    if not positions:
        return

    do_api_check = (time.time() - state.get("last_settle_check", 0)) >= SETTLE_CHECK_INTERVAL
    if do_api_check:
        state["last_settle_check"] = time.time()

    sem = asyncio.Semaphore(5)

    async def check_one(pos):
        async with sem:
            event  = pos["event"]
            sets   = float(pos["sets"] or 1.0)
            cost   = float(pos["cost"])

            # Pass 1 — real resolution via Kalshi API (throttled to once/15 min)
            if do_api_check:
                try:
                    data    = await _get(client, "/markets", {"event_ticker": event, "limit": 50})
                    markets = data.get("markets", [])
                    winner  = next((m for m in markets if m.get("result") == "yes"), None)
                    if winner:
                        payout = round(sets * 1.0, 4)
                        if settle_position(event, payout):
                            log(f"  [PAPER] settled (resolved): {event}  pnl=${payout - cost:+.4f}", "success")
                            notify_close(event, pos["platform"], cost, payout)
                            log_close(pos["platform"], event, sets, cost, payout)
                        return
                    # Void check: all legs finalized but no winner → Kalshi voided, refund at cost
                    if markets and all(m.get("status") in ("finalized", "settled") for m in markets):
                        if settle_position(event, cost):
                            log(f"  [PAPER] voided: {event} — refunded ${cost:.4f} (no winner)", "warn")
                            notify_close(event, pos["platform"], cost, cost)
                            log_close(pos["platform"], event, sets, cost, cost)
                        return
                except Exception:
                    pass

            # Stuck-position warning: open far longer than expected close window
            age_days = (datetime.now() - datetime.fromisoformat(pos["ts"])).total_seconds() / 86400
            max_days = float(os.getenv("MAX_DAYS_TO_CLOSE", "7"))
            if age_days > max_days + 3:
                log(f"  [PAPER] WARNING: {event} open {age_days:.1f}d — may be stuck (no resolution yet)", "warn")

            # Pass 2 — simulated settlement after SIM_SETTLE_HOURS
            if SIM_SETTLE_HOURS > 0:
                age_h = (datetime.now() - datetime.fromisoformat(pos["ts"])).total_seconds() / 3600
                if age_h >= SIM_SETTLE_HOURS:
                    payout = round(cost + float(pos["expected_profit"]), 4)
                    if settle_position(event, payout):
                        log(f"  [PAPER] settled (sim {age_h:.1f}h): {event}  pnl=+${pos['expected_profit']:.4f}", "success")
                        notify_close(event, pos["platform"], cost, payout)
                        log_close(pos["platform"], event, sets, cost, payout)

    await asyncio.gather(*[check_one(p) for p in positions])


async def run_scan():
    # Circuit breaker: halt trading if daily loss exceeds threshold
    daily_loss, _ = daily_pnl()
    if daily_loss < -MAX_DAILY_LOSS:
        log(f"⚠ CIRCUIT BREAKER: daily P&L ${daily_loss:.2f} exceeds -${MAX_DAILY_LOSS:.2f} limit — scan only, no trades", "error")
        state["circuit_open"] = True
    else:
        state["circuit_open"] = False

    state["scanning"] = True
    t0 = time.time()
    log("── scan started ──", "system")
    await broadcast("scan_start", {"ts": ts()})

    async with httpx.AsyncClient() as client:
        kalshi = KalshiClient()

        # Settle any resolved paper positions before scanning for new ones
        await settle_open_positions(client)

        # Balance
        try:
            bal = await kalshi.get_balance(client)
            state["balance"] = bal
            log(f"balance: ${bal:.2f}", "info")
        except Exception as e:
            log(f"balance error: {e}", "error")

        # Kalshi MECE
        log("scanning kalshi mece markets...", "dim")
        try:
            opps = await kalshi.scan_mece_opportunities(
                client, min_profit=MIN_PROFIT, max_profit=MAX_PROFIT
            )
            state["kalshi_opps"] = [
                {
                    "ticker":       o["event_ticker"],
                    "title":        o["title"],
                    "profit":       round(o["profit"], 4),
                    "gross_profit": round(o["gross_profit"], 4),
                    "fee_per_set":  round(o["fee_per_set"], 4),
                    "min_size":     round(o["min_size"], 2),
                    "book_sets":    o.get("book_max_sets", 0),
                    "deploy":       round(o.get("book_total_cost", 0), 2),
                    "exp_profit":   round(o.get("book_total_profit", 0), 4),
                    "volume":       round(o["total_volume"], 0),
                    "safe":         o["has_catchall"],
                    "n_markets":    o["n_markets"],
                    "days_to_close": o.get("days_to_close", "?"),
                }
                for o in opps
            ]
            log(f"mece scan: {len(opps)} opportunities found", "info" if opps else "dim")
            for o in opps[:3]:
                flag = "SAFE" if o["has_catchall"] else "INCOMPLETE"
                sets = o.get("book_max_sets", 0)
                deploy = o.get("book_total_cost", 0)
                ep = o.get("book_total_profit", 0)
                log(f"  {o['event_ticker']}  {o['profit']*100:.1f}% net  {sets} sets  ${deploy:.2f} → +${ep:.2f}  [{flag}]",
                    "success" if o["has_catchall"] else "warn")
            # Paper trade all profitable MECE arbs (safe and incomplete)
            if DRY_RUN and not state.get("circuit_open"):
                for o in opps:
                    sets   = o.get("book_max_sets", 0)
                    cost   = o.get("book_total_cost", 0.0)
                    profit = o.get("book_total_profit", 0.0)
                    if sets <= 0 or cost <= 0:
                        continue
                    pt = record_paper_trade(
                        platform="Kalshi",
                        event=o["event_ticker"],
                        sets=sets,
                        cost=round(cost, 6),
                        expected_profit=round(profit, 6),
                        orders=[{"ticker": m.get("ticker"), "at_price": float(m.get("yes_ask_dollars", 0))}
                                for m in o.get("markets", [])],
                    )
                    if pt["status"] == "opened":
                        flag = "SAFE" if o["has_catchall"] else "INCOMPLETE"
                        log(f"  [PAPER] opened [{flag}]: {o['event_ticker']}  ${cost:.2f}  ep=+${profit:.2f}", "success")
                        notify_open(o["event_ticker"], "Kalshi", sets, cost, profit)
                        log_open("Kalshi", o["event_ticker"], sets, cost, profit)
                    elif pt["status"] == "duplicate":
                        log(f"  [PAPER] holding: {o['event_ticker']}", "dim")
        except Exception as e:
            log(f"mece scan error: {e}", "error")

        # Binary cross-platform — reuse MECE opps already fetched above
        log("scanning binary cross-platform arb...", "dim")
        try:
            res = await scan_binary_arb(
                client,
                min_profit=MIN_PROFIT,
                mece_opps=opps,
                max_position_usdc=float(os.getenv("MAX_POSITION_USDC", "50")),
            )
            pairs = [
                {
                    "score":    km_pm_sc[2],
                    "k_title":  km_pm_sc[0]["title"],
                    "p_title":  km_pm_sc[1]["question"],
                    "k_yes":    km_pm_sc[0]["yes_ask"],
                    "p_yes":    km_pm_sc[1]["yes_snap"],
                    "k_no":     km_pm_sc[0]["no_ask"],
                    "p_no":     km_pm_sc[1]["no_snap"],
                    "gap":      round(max(
                        1 - km_pm_sc[0]["yes_ask"] - km_pm_sc[1]["no_snap"],
                        1 - km_pm_sc[1]["yes_snap"] - km_pm_sc[0]["no_ask"],
                    ), 4),
                    "dir":      ("K_YES+P_NO"
                                 if 1 - km_pm_sc[0]["yes_ask"] - km_pm_sc[1]["no_snap"] >=
                                    1 - km_pm_sc[1]["yes_snap"] - km_pm_sc[0]["no_ask"]
                                 else "P_YES+K_NO"),
                }
                for km_pm_sc in res["candidates"][:20]
            ]
            arbs = [
                {
                    "ticker":    a.kalshi_ticker,
                    "k_title":   a.kalshi_title,
                    "p_title":   a.poly_question,
                    "profit":    round(a.profit_pct, 4),
                    "direction": a.direction,
                    "k_yes":     a.k_yes_ask,
                    "p_yes":     a.p_yes_ask,
                    "k_no":      a.k_no_ask,
                    "p_no":      a.p_no_ask,
                }
                for a in res["opportunities"]
            ]
            state["binary"] = {
                "kalshi_count": res["kalshi_count"],
                "poly_count":   res["poly_count"],
                "pair_count":   len(res["candidates"]),
                "pairs":        pairs,
                "arbs":         arbs,
            }
            log(f"binary scan: {len(res['candidates'])} pairs · {res['kalshi_count']} kalshi · {res['poly_count']} poly", "info")
            if arbs:
                for a_dict, a_obj in zip(arbs, res["opportunities"]):
                    msg = f"*** ARB FOUND: {a_dict['profit']*100:.2f}% — {a_dict['k_title'][:45]}"
                    log(msg, "arb")
                    state["arb_alerts"].appendleft(
                        {"ts": ts(), "profit": a_dict["profit"], "k_title": a_dict["k_title"],
                         "p_title": a_dict["p_title"], "direction": a_dict["direction"]}
                    )
                    if DRY_RUN:
                        # Fall back to 1 contract if live book depth was unavailable
                        paper_contracts = a_obj.max_contracts if a_obj.max_contracts > 0 else 1.0
                        cost   = (a_obj.kalshi_leg + a_obj.poly_leg) * paper_contracts
                        profit = a_obj.profit_pct * paper_contracts
                        k_side = "yes" if a_obj.direction == "buy_kalshi_yes_poly_no" else "no"
                        p_side = "no"  if a_obj.direction == "buy_kalshi_yes_poly_no" else "yes"
                        pt = record_paper_trade(
                            platform="BinaryArb",
                            event=a_obj.kalshi_ticker,
                            sets=paper_contracts,
                            cost=round(cost, 6),
                            expected_profit=round(profit, 6),
                            orders=[
                                {"ticker": a_obj.kalshi_ticker,
                                 "side": f"kalshi_{k_side}",
                                 "at_price": a_obj.kalshi_leg},
                                {"ticker": a_obj.poly_market_id,
                                 "side": f"poly_{p_side}",
                                 "at_price": a_obj.poly_leg},
                            ],
                        )
                        if pt["status"] == "opened":
                            log(f"  [PAPER] binary arb opened: {a_obj.kalshi_ticker}  cost=${cost:.4f}  ep=${profit:.4f}", "success")
                            notify_open(a_obj.kalshi_ticker, "BinaryArb", paper_contracts, cost, profit)
                            log_open("BinaryArb", a_obj.kalshi_ticker, paper_contracts, cost, profit)
            else:
                best = res["candidates"][0] if res["candidates"] else None
                if best:
                    km, pm, sc = best
                    gap = max(1 - km["yes_ask"] - pm["no_snap"],
                              1 - pm["yes_snap"] - km["no_ask"])
                    log(f"  best gap: {gap*100:+.1f}% [{sc:.2f}] {km['title'][:40]}", "dim")
        except Exception as e:
            log(f"binary scan error: {e}", "error")

    dur = round(time.time() - t0, 1)
    state["scanning"]      = False
    state["scan_count"]   += 1
    state["last_scan_ts"]  = ts()
    state["last_scan_dur"] = dur
    if dur > SCAN_INTERVAL * 0.8:
        log(f"── scan done in {dur}s (slow — interval is {SCAN_INTERVAL}s) ──", "warn")
    else:
        log(f"── scan done in {dur}s ──", "system")

    pnl, trades = daily_pnl()
    paper = get_paper_summary()
    state["paper"] = paper

    # P&L change notification — fire when realized P&L moves
    prev_realized = state.get("_last_realized_pnl")
    cur_realized  = paper.get("realized_pnl", 0.0)
    if prev_realized is not None and abs(cur_realized - prev_realized) >= 0.01:
        notify_pnl(cur_realized, cur_realized - prev_realized, paper.get("settled_count", 0))
    state["_last_realized_pnl"] = cur_realized

    # Midnight rollover — save yesterday's summary when the day changes
    today = datetime.now().strftime("%Y-%m-%d")
    if today != state.get("_current_day"):
        yesterday = state["_current_day"]
        state["_current_day"] = today
        try:
            report = generate_summary(yesterday)
            log(f"daily summary saved → logs/summary_{yesterday}.txt", "system")
            notify_pnl(cur_realized, 0, paper.get("settled_count", 0))
        except Exception as e:
            log(f"summary generation failed: {e}", "error")

    await broadcast("state_update", build_state_payload(pnl, trades))


def build_state_payload(pnl=None, trades=None) -> dict:
    if pnl is None:
        pnl, trades = daily_pnl()
    uptime = int(time.time() - state["start_time"])
    h, r = divmod(uptime, 3600); m, s = divmod(r, 60)
    return {
        "mode":          state["mode"],
        "balance":       state["balance"],
        "scan_count":    state["scan_count"],
        "uptime":        f"{h:02d}:{m:02d}:{s:02d}",
        "last_scan_ts":  state["last_scan_ts"],
        "last_scan_dur": state["last_scan_dur"],
        "scanning":      state["scanning"],
        "pnl":           round(pnl, 4),
        "trades":        trades,
        "kalshi_opps":   state["kalshi_opps"],
        "binary":        state["binary"],
        "log":           list(state["log"])[-40:],
        "arb_alerts":    list(state["arb_alerts"]),
        "next_scan_in":  state["next_scan_in"],
        "paper":         state.get("paper") or get_paper_summary(),
    }


async def scan_loop():
    await asyncio.sleep(1)
    while True:
        try:
            await run_scan()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Scan loop crashed — recovering in 30s")
            log(f"CRITICAL: scan crashed: {e} — resuming in 30s", "error")
            await asyncio.sleep(30)
            continue
        for remaining in range(SCAN_INTERVAL, 0, -1):
            state["next_scan_in"] = remaining
            await asyncio.sleep(1)
            # Tick heartbeat every 5s so the countdown updates
            if remaining % 5 == 0:
                pnl, trades = daily_pnl()
                await broadcast("tick", {
                    "scanning":     False,
                    "next_scan_in": remaining,
                    "uptime":       _uptime_str(),
                    "pnl":          round(pnl, 4),
                    "trades":       trades,
                    "paper":        state.get("paper"),
                })
        state["next_scan_in"] = 0


def _uptime_str() -> str:
    uptime = int(time.time() - state["start_time"])
    h, r = divmod(uptime, 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    asyncio.create_task(scan_loop())
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html = (Path(__file__).parent / "static" / "index.html").read_text()
    return HTMLResponse(html)


@app.get("/api/state")
async def api_state():
    return build_state_payload()


@app.get("/api/daily-summary")
async def api_daily_summary(date: str = None):
    from fastapi.responses import PlainTextResponse
    try:
        report = generate_summary(date)
        return PlainTextResponse(report)
    except Exception as e:
        return PlainTextResponse(f"Error generating summary: {e}", status_code=500)


@app.get("/api/tx-log")
async def api_tx_log(date: str = None):
    from fastapi.responses import FileResponse
    from utils.daily_logger import get_tx_log_path
    path = get_tx_log_path(date)
    if not path.exists():
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("No transaction log for that date.", status_code=404)
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    # Send current state immediately on connect
    pnl, trades = daily_pnl()
    await ws.send_text(json.dumps({
        "event": "state_update",
        "data":  build_state_payload(pnl, trades)
    }))
    try:
        while True:
            await ws.receive_text()   # keep alive, client pings
    except WebSocketDisconnect:
        clients.discard(ws)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print(f"\n  ARB BOT DASHBOARD  →  http://localhost:{PORT}\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
