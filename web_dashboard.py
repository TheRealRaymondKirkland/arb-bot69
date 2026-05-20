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
from sports.sports_scanner import scan_sports_arb
from weather.weather_scanner import scan_weather_opportunities
from utils.trade_log import daily_pnl
from utils.paper_trader import get_paper_summary, record_paper_trade, get_open_positions, settle_position
from utils.notifier import notify_open, notify_close, notify_pnl
from utils.daily_logger import log_open, log_close, generate_summary

logging.basicConfig(level=logging.INFO)

SCAN_INTERVAL         = int(os.getenv("SCAN_INTERVAL", "120"))
WEATHER_SCAN_INTERVAL = int(os.getenv("WEATHER_SCAN_INTERVAL", "30"))
DRY_RUN               = os.getenv("DRY_RUN", "true").lower() != "false"
MIN_PROFIT            = float(os.getenv("MIN_PROFIT", "0.03"))
MAX_PROFIT            = float(os.getenv("MAX_PROFIT", "0.15"))
MAX_DAILY_LOSS        = float(os.getenv("MAX_DAILY_LOSS", "20"))
PORT                  = int(os.getenv("DASHBOARD_PORT", "8080"))
MAX_POSITION_USDC     = float(os.getenv("MAX_POSITION_USDC", "50"))
SIM_SETTLE_HOURS      = float(os.getenv("SIM_SETTLE_HOURS", "0"))
SETTLE_CHECK_INTERVAL = 900   # Poll Kalshi API for real resolution every 15 min

# ─── Shared state ─────────────────────────────────────────────────────────────

state = {
    "mode":              "DRY RUN" if DRY_RUN else "LIVE",
    "balance":           None,
    "scan_count":        0,
    "wx_scan_count":     0,
    "start_time":        time.time(),
    "last_scan_ts":      None,
    "last_scan_dur":     None,
    "last_wx_scan_ts":   None,
    "last_wx_scan_dur":  None,
    "scanning":          False,
    "wx_scanning":       False,
    "kalshi_opps":       [],
    "binary":            None,
    "sports":            None,
    "weather":           None,
    "log":               deque(maxlen=80),
    "arb_alerts":        deque(maxlen=20),
    "next_scan_in":      SCAN_INTERVAL,
    "next_wx_in":        WEATHER_SCAN_INTERVAL,
    "paper":             None,
    "_current_day":      datetime.now().strftime("%Y-%m-%d"),
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
    Settle open paper positions by querying Kalshi API.
    MECE arb: any leg with result='yes' → win (exactly one always wins).
    WeatherEdge: check our specific ticker — win/loss based on action vs result.
    """
    positions = get_open_positions()
    if not positions:
        return

    do_api_check = (time.time() - state.get("last_settle_check", 0)) >= SETTLE_CHECK_INTERVAL
    if do_api_check:
        state["last_settle_check"] = time.time()
    if not do_api_check:
        return

    sem = asyncio.Semaphore(5)

    async def settle_mece(pos):
        """MECE arb: fetch all legs of the event, find winner."""
        async with sem:
            event = pos["event"]
            sets  = float(pos["sets"] or 1.0)
            cost  = float(pos["cost"])
            try:
                data    = await _get(client, "/markets", {"event_ticker": event, "limit": 50})
                markets = data.get("markets", [])
                winner  = next((m for m in markets if m.get("result") == "yes"), None)
                if winner:
                    payout = round(sets * 1.0, 4)
                    if settle_position(event, payout):
                        log(f"  [PAPER] settled: {event}  pnl=${payout-cost:+.4f}", "success")
                        notify_close(event, pos["platform"], cost, payout)
                        log_close(pos["platform"], event, sets, cost, payout)
                    return
                if markets and all(m.get("status") in ("finalized", "settled") for m in markets):
                    if settle_position(event, cost):
                        log(f"  [PAPER] voided: {event} — refund ${cost:.4f}", "warn")
                        notify_close(event, pos["platform"], cost, cost)
                        log_close(pos["platform"], event, sets, cost, cost)
            except Exception:
                pass
            # Stuck warning
            age_days = (datetime.now() - datetime.fromisoformat(pos["ts"])).total_seconds() / 86400
            if age_days > float(os.getenv("MAX_DAYS_TO_CLOSE", "30")) + 3:
                log(f"  [PAPER] WARNING: {event} open {age_days:.1f}d — possibly stuck", "warn")

    async def settle_weather(pos):
        """WeatherEdge: resolve by checking our specific market ticker."""
        async with sem:
            market_ticker  = pos["event"]      # stored as market ticker for WeatherEdge
            sets  = float(pos["sets"] or 1.0)
            cost  = float(pos["cost"])
            try:
                orders = json.loads(pos.get("orders") or "[]")
            except Exception:
                orders = []
            action       = orders[0].get("action", "buy_yes") if orders else "buy_yes"
            event_ticker = orders[0].get("event_ticker", "") if orders else ""

            if not event_ticker:
                return

            try:
                data    = await _get(client, "/markets", {
                    "event_ticker": event_ticker, "limit": 20
                })
                markets = data.get("markets", [])
                # Find the winning bucket
                winner = next((m for m in markets if m.get("result") == "yes"), None)
                all_done = markets and all(
                    m.get("status") in ("finalized", "settled") for m in markets
                )

                if winner is None and not all_done:
                    return   # still live

                if winner is None and all_done:
                    # Voided event
                    if settle_position(market_ticker, cost):
                        log(f"  [PAPER] weather VOIDED: {market_ticker} — refund ${cost:.4f}", "warn")
                    return

                # Market resolved — check if we're on the right side
                winner_ticker = winner.get("ticker", "") if winner else ""
                we_won = (
                    (action == "buy_yes" and winner_ticker == market_ticker) or
                    (action == "buy_no"  and winner_ticker != market_ticker)
                )
                payout = round(sets * 1.0, 4) if we_won else 0.0
                if settle_position(market_ticker, payout):
                    pnl = payout - cost
                    if we_won:
                        log(f"  [PAPER] weather WIN ({action}): {market_ticker}  pnl=${pnl:+.4f}", "success")
                    else:
                        log(f"  [PAPER] weather LOSS ({action}): {market_ticker}  pnl=${pnl:+.4f}", "warn")
                    notify_close(market_ticker, "WeatherEdge", cost, payout)
                    log_close("WeatherEdge", market_ticker, sets, cost, payout)
            except Exception as e:
                logger.debug(f"WeatherEdge settle failed for {market_ticker}: {e}")

        # Stuck warning
        age_days = (datetime.now() - datetime.fromisoformat(pos["ts"])).total_seconds() / 86400
        if age_days > 5:
            log(f"  [PAPER] WARNING: weather {market_ticker} open {age_days:.1f}d — may be stuck", "warn")

    tasks = []
    for pos in positions:
        if pos["platform"] == "WeatherEdge":
            tasks.append(settle_weather(pos))
        elif pos["platform"] == "Kalshi":
            tasks.append(settle_mece(pos))

    if tasks:
        await asyncio.gather(*tasks)


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

        # Sports arb scan
        log("scanning sports arb (Kalshi × Polymarket)...", "dim")
        try:
            sports = await scan_sports_arb(client, min_profit=MIN_PROFIT)
            pairs = [
                {
                    "team":          p.team,
                    "sport":         p.sport,
                    "k_yes":         p.kalshi_yes_ask,
                    "k_bid":         p.kalshi_yes_bid,
                    "p_yes":         p.poly_yes_ask,
                    "p_bid":         p.poly_yes_bid,
                    "gap":           round(p.gap, 4),
                    "direction":     p.direction,
                    "k_ticker":      p.kalshi_ticker,
                    "poly_question": p.poly_question,
                }
                for p in sports["pairs"]
            ]
            arbs = [
                {
                    "team":      a.team,
                    "sport":     a.sport,
                    "direction": a.direction,
                    "profit":    a.profit_pct,
                    "gross":     a.gross_profit,
                    "k_yes":     a.kalshi_yes_ask,
                    "p_yes":     a.poly_yes_ask,
                    "k_ticker":  a.kalshi_ticker,
                }
                for a in sports["opportunities"]
            ]
            state["sports"] = {
                "kalshi_count": sports["kalshi_count"],
                "poly_count":   sports["poly_count"],
                "pair_count":   len(pairs),
                "pairs":        pairs,
                "arbs":         arbs,
            }
            log(f"sports scan: {len(pairs)} matched pairs · {len(arbs)} arbs", "info" if arbs else "dim")
            for a in arbs:
                msg = f"*** SPORTS ARB: {a['profit']*100:.2f}% — {a['team']} ({a['sport']}) [{a['direction']}]"
                log(msg, "arb")
                state["arb_alerts"].appendleft({
                    "ts": ts(), "profit": a["profit"], "k_title": a["team"],
                    "p_title": a["sport"], "direction": a["direction"]
                })
                if DRY_RUN and not state.get("circuit_open"):
                    cost   = round(1 - a["profit"], 6)
                    profit = round(a["profit"], 6)
                    pt = record_paper_trade(
                        platform="SportsArb",
                        event=a["k_ticker"],
                        sets=1,
                        cost=cost,
                        expected_profit=profit,
                        orders=[{"ticker": a["k_ticker"], "direction": a["direction"],
                                 "k_yes": a["k_yes"], "p_yes": a["p_yes"]}],
                    )
                    if pt["status"] == "opened":
                        log(f"  [PAPER] sports arb opened: {a['k_ticker']}  cost=${cost:.4f}  ep=+${profit:.4f}", "success")
                        notify_open(a["k_ticker"], f"SportsArb/{a['sport']}", 1, cost, profit)
                        log_open("SportsArb", a["k_ticker"], 1, cost, profit)
                    elif pt["status"] == "duplicate":
                        log(f"  [PAPER] holding: {a['k_ticker']}", "dim")
        except Exception as e:
            log(f"sports scan error: {e}", "error")

        # Weather scan runs in its own fast loop (see weather_scan_loop below)

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
        "sports":        state["sports"],
        "weather":       state["weather"],
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


async def run_weather_scan():
    """Fast standalone weather scan — runs every WEATHER_SCAN_INTERVAL seconds."""
    state["wx_scanning"] = True
    t0 = time.time()

    paper   = state.get("paper") or get_paper_summary()
    balance = float(paper.get("virtual_balance") or 1000.0)

    async with httpx.AsyncClient() as client:
        # Also run settlement on WeatherEdge positions
        wx_positions = [p for p in get_open_positions() if p["platform"] == "WeatherEdge"]
        if wx_positions:
            await settle_open_positions(client)

        try:
            wx   = await scan_weather_opportunities(
                client,
                max_days=3.0,
                portfolio_balance=balance,
            )
            opps = wx["opportunities"]
            state["weather"] = {
                "event_count":   wx["event_count"],
                "pair_count":    len(wx["all_pairs"]),
                "opp_count":     len(opps),
                "ensemble_pct":  wx.get("ensemble_pct", 0),
                "opportunities": [
                    {
                        "event":      o.event_ticker,
                        "city":       o.city.title(),
                        "metric":     o.metric,
                        "date":       o.target_date,
                        "leg":        o.leg_title,
                        "kalshi_ask": o.kalshi_ask,
                        "noaa_prob":  o.prob,
                        "noaa_temp":  o.forecast,
                        "edge":       o.edge,
                        "net_profit": o.net_profit,
                        "action":     o.action,
                        "days_left":  o.days_to_close,
                        "contracts":  o.contracts,
                        "deploy_usd": o.deploy_usd,
                    }
                    for o in opps
                ],
                "all_pairs": wx["all_pairs"],
            }

            ens_pct = wx.get("ensemble_pct", 0)
            src     = f"ensemble {ens_pct}%" if ens_pct > 0 else "NOAA+Gaussian"
            log(
                f"wx [{src}]: {wx['event_count']} events · "
                f"{len(wx['all_pairs'])} legs · {len(opps)} edges",
                "info" if opps else "dim",
            )

            for o in opps:
                direction = "BUY YES" if o.action == "buy_yes" else "BUY NO"
                msg = (
                    f"*** WEATHER EDGE: {o.edge*100:+.1f}% — "
                    f"{o.city.title()} {o.metric} {o.leg_title}  "
                    f"forecast={o.forecast}°F  {o.contracts}x [{direction}]  "
                    f"deploy=${o.deploy_usd:.2f}"
                )
                log(msg, "arb")
                state["arb_alerts"].appendleft({
                    "ts":        ts(),
                    "profit":    abs(o.net_profit),
                    "k_title":   f"{o.city.title()} {o.metric} {o.leg_title}",
                    "p_title":   f"forecast={o.forecast}°F  Kelly={o.contracts}x",
                    "direction": direction,
                })

                if DRY_RUN and not state.get("circuit_open"):
                    cost   = o.deploy_usd
                    profit = round(o.net_profit, 6)
                    pt = record_paper_trade(
                        platform="WeatherEdge",
                        event=o.ticker,
                        sets=o.contracts,
                        cost=round(cost, 6),
                        expected_profit=profit,
                        orders=[{
                            "ticker":       o.ticker,
                            "event_ticker": o.event_ticker,
                            "action":       o.action,
                            "at_price":     o.kalshi_ask,
                            "contracts":    o.contracts,
                        }],
                    )
                    if pt["status"] == "opened":
                        log(
                            f"  [PAPER] wx trade: {o.ticker}  {direction}  "
                            f"{o.contracts}x  cost=${cost:.2f}  ep=+${profit:.4f}",
                            "success",
                        )
                        notify_open(o.ticker, "WeatherEdge", o.contracts, cost, profit)
                        log_open("WeatherEdge", o.ticker, o.contracts, cost, profit)

        except Exception as e:
            log(f"weather scan error: {e}", "error")

    dur = round(time.time() - t0, 1)
    state["wx_scanning"]      = False
    state["wx_scan_count"]   += 1
    state["last_wx_scan_ts"]  = ts()
    state["last_wx_scan_dur"] = dur

    # Update paper state and broadcast weather update
    state["paper"] = get_paper_summary()
    pnl, trades = daily_pnl()
    await broadcast("state_update", build_state_payload(pnl, trades))


async def weather_scan_loop():
    """Dedicated fast weather scan loop — independent of main scan cadence."""
    from weather.noaa_client import prefetch_all as noaa_prefetch
    from weather.openmeteo_client import prefetch_all as om_prefetch
    await asyncio.sleep(3)   # let the main loop start first
    # Warm both forecast caches in parallel before first scan
    try:
        await asyncio.gather(noaa_prefetch(), om_prefetch(), return_exceptions=True)
        log("NOAA + Open-Meteo ensemble pre-cached for all cities", "system")
    except Exception as e:
        log(f"forecast prefetch warning: {e}", "warn")

    while True:
        try:
            await run_weather_scan()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Weather scan loop crashed")
            log(f"wx crash: {e} — resuming in {WEATHER_SCAN_INTERVAL}s", "error")
        await asyncio.sleep(WEATHER_SCAN_INTERVAL)


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    asyncio.create_task(scan_loop())
    asyncio.create_task(weather_scan_loop())
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
