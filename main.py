#!/home/raykirkland88/poly-trading-bot/venv/bin/python
"""
Kalshi/Polymarket Arbitrage Bot

Strategies:
  1. Kalshi MECE arb — buy all YES outcomes in mutually-exclusive markets
     where sum(YES ask) < $1.00. Risk-free if the market is complete.
  2. Polymarket neg-risk arb — same idea inside Polymarket's neg-risk events.
  3. Cross-platform scan — compare prices for the same event across both
     platforms (report only; no auto-execution without verified pairs).
  4. Binary cross-platform arb — find the same YES/NO question priced
     differently on Kalshi vs Polymarket and buy both sides for profit.

Run:
  python main.py               # scan + report, DRY_RUN=true (safe)
  DRY_RUN=false python main.py # live execution (real money)
  python main.py --once        # scan once and exit (no loop)
  python main.py --cross       # include legacy cross-platform scan
  python main.py --binary      # binary cross-platform arb scan (new)
  python main.py --binary --once  # single binary scan and exit
"""
import asyncio
import logging
import os
import sys
import time
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

import httpx
from dotenv import load_dotenv

load_dotenv()

from kalshi.kalshi_client import KalshiClient
from polymarket.polymarket_client import PolymarketClient
from cross_platform.scanner import CrossPlatformScanner, print_cross_platform_report
from utils.trade_log import daily_pnl, log_trade, print_daily_summary
from utils.paper_trader import record_paper_trade, get_paper_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DRY_RUN = os.getenv("DRY_RUN", "true").lower() != "false"
MIN_PROFIT = float(os.getenv("MIN_PROFIT", "0.04"))
MAX_POSITION_USDC = float(os.getenv("MAX_POSITION_USDC", "50"))
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "60"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "20"))
RUN_CROSS  = "--cross"  in sys.argv
RUN_BINARY = "--binary" in sys.argv


def banner():
    mode = "DRY RUN (no real orders)" if DRY_RUN else "LIVE TRADING — REAL MONEY"
    print(f"""
╔══════════════════════════════════════════════════════════╗
║         KALSHI / POLYMARKET ARB BOT                     ║
║  Mode:   {mode:<46} ║
║  Min profit threshold: {MIN_PROFIT*100:.1f}%                            ║
║  Max per trade: ${MAX_POSITION_USDC:<41.0f} ║
║  Circuit breaker: -${MAX_DAILY_LOSS:<39.0f} ║
╚══════════════════════════════════════════════════════════╝""")


def print_opportunity(opp: dict, platform: str, rank: int):
    safe = "✅ COMPLETE" if opp.get("has_catchall") else "⚠️  INCOMPLETE"
    print(f"\n  #{rank} [{platform}] {opp.get('event_ticker') or opp.get('event_slug', '')}")
    print(f"     Title    : {opp.get('title', '')[:60]}")
    print(f"     Markets  : {opp['n_markets']} candidates")
    print(f"     Sum ask  : ${opp['sum_yes_ask']:.4f}")
    print(f"     Profit   : ${opp['profit']:.4f}  ({opp['profit']*100:.2f}%)")
    print(f"     Min size : {opp['min_size']:.2f} contracts")
    print(f"     Max earn : ${opp['profit'] * opp['min_size']:.2f} at full size")
    print(f"     Volume   : ${opp['total_volume']:,.0f}")
    if platform == "Kalshi":
        print(f"     Safety   : {safe}")
        print(f"     Candidates:")
        for m in opp["markets"]:
            print(f"       {m.get('yes_sub_title','?')[:35]:<35} ask=${float(m.get('yes_ask_dollars',0)):.3f} size={float(m.get('yes_ask_size_fp',0)):.1f}")


async def scan_once(kalshi: KalshiClient, poly: PolymarketClient):
    print(f"\n{'='*60}")
    print(f"  Scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    async with httpx.AsyncClient() as client:
        # --- Kalshi balance ---
        try:
            bal = await kalshi.get_balance(client)
            print(f"\n  Kalshi balance: ${bal:.2f}")
        except Exception as e:
            print(f"\n  Kalshi balance: ERROR ({e})")
            bal = 0

        # --- Kalshi MECE scan ---
        print(f"\n  Scanning Kalshi MECE markets (min profit {MIN_PROFIT*100:.0f}%)...")
        kalshi_opps = await kalshi.scan_mece_opportunities(client, min_profit=MIN_PROFIT)
        print(f"  Found {len(kalshi_opps)} Kalshi arb opportunities")

        for i, opp in enumerate(kalshi_opps[:5], 1):
            print_opportunity(opp, "Kalshi", i)

        # --- Circuit breaker check ---
        pnl_today, trades_today = daily_pnl()
        print_daily_summary()
        if pnl_today < -MAX_DAILY_LOSS:
            logger.error(
                f"CIRCUIT BREAKER: daily P&L ${pnl_today:.2f} < -${MAX_DAILY_LOSS:.2f}. "
                "Scanning only — execution halted."
            )
            continue_execution = False
        else:
            continue_execution = True

        # --- Execute Kalshi arbs ---
        for opp in kalshi_opps:
            if not opp["has_catchall"]:
                logger.info(f"Skipping INCOMPLETE market {opp['event_ticker']} — requires manual review")
                continue
            if opp["min_size"] < 0.1:
                logger.info(f"Skipping {opp['event_ticker']}: min_size too small ({opp['min_size']:.3f})")
                continue
            if not continue_execution:
                logger.warning(f"Circuit breaker active — skipping execution of {opp['event_ticker']}")
                continue

            result = await kalshi.execute_mece_arb(client, opp, max_usdc=MAX_POSITION_USDC)
            status = result["status"]
            cost = result.get("total_cost", 0)
            profit = result.get("expected_profit", 0)
            print(f"\n  >> [{status.upper()}] {opp['event_ticker']}: cost=${cost:.2f} expected_profit=${profit:.2f}")
            log_trade(
                platform="Kalshi",
                event=opp["event_ticker"],
                sets=result.get("sets", 0),
                cost=cost,
                expected_profit=profit,
                status=status,
                orders=result.get("orders", []),
            )
            if DRY_RUN and status == "dry_run" and cost > 0:
                pt = record_paper_trade(
                    platform="Kalshi",
                    event=opp["event_ticker"],
                    sets=result.get("sets", 0),
                    cost=cost,
                    expected_profit=profit,
                    orders=result.get("orders", []),
                )
                print(f"     [PAPER] {pt['status'].upper()}: {opp['event_ticker']}")
                if pt["status"] == "opened":
                    ps = get_paper_summary()
                    print(f"     [PAPER] Virtual balance: ${ps['virtual_balance']:.2f} | "
                          f"Deployed: ${ps['deployed']:.2f} | Expected P&L: ${ps['open_ep']:.4f}")

        # --- Polymarket neg-risk scan ---
        print(f"\n  Scanning Polymarket neg-risk markets...")
        try:
            poly_opps = await poly.get_neg_risk_opportunities(client, min_profit=MIN_PROFIT)
            print(f"  Found {len(poly_opps)} Polymarket arb opportunities")

            for i, opp in enumerate(poly_opps[:3], 1):
                print_opportunity(opp, "Polymarket", i)

            for opp in poly_opps:
                if opp["min_size"] < 1.0:
                    continue
                if not continue_execution:
                    logger.warning(f"Circuit breaker active — skipping Poly execution of {opp['title'][:40]}")
                    continue
                result = await poly.execute_neg_risk_arb(opp, max_usdc=MAX_POSITION_USDC)
                status = result["status"]
                cost = result.get("total_cost", 0)
                profit = result.get("expected_profit", 0)
                print(f"\n  >> [{status.upper()}] {opp['title'][:40]}: cost=${cost:.2f} expected_profit=${profit:.2f}")
                log_trade(
                    platform="Polymarket",
                    event=opp.get("event_slug", opp["title"][:40]),
                    sets=result.get("sets", 0),
                    cost=cost,
                    expected_profit=profit,
                    status=status,
                    orders=result.get("orders", []),
                )
        except Exception as e:
            print(f"  Polymarket scan failed: {e}")

        # --- Binary cross-platform arb (--binary flag) ---
        if RUN_BINARY:
            print(f"\n  Running binary cross-platform scan (Kalshi ↔ Polymarket)...")
            try:
                from cross_platform.binary_scanner import scan_binary_arb, print_binary_report
                # Pass existing kalshi_opps so binary scanner doesn't re-run MECE scan
                binary_results = await scan_binary_arb(
                    client,
                    mece_opps=kalshi_opps,
                    max_position_usdc=MAX_POSITION_USDC,
                )
                print_binary_report(binary_results)
            except Exception as e:
                print(f"  Binary scan failed: {e}")
                logger.exception("Binary scan error")

        # --- Cross-platform scan (--cross flag only — slower) ---
        if RUN_CROSS:
            print(f"\n  Running cross-platform scan (Kalshi ↔ Polymarket)...")
            try:
                cross_scanner = CrossPlatformScanner(min_spread=MIN_PROFIT)
                # Pass existing kalshi_opps to avoid re-running the Kalshi scan
                cross_results = await cross_scanner.scan(client, kalshi_opps=kalshi_opps)
                print_cross_platform_report(cross_results)
            except Exception as e:
                print(f"  Cross-platform scan failed: {e}")


async def main(run_once: bool = False):
    banner()

    if not DRY_RUN:
        print("\n  ⚠️  LIVE MODE — real money will be traded.")
        confirm = input("  Type YES to confirm: ")
        if confirm.strip() != "YES":
            print("  Aborted.")
            return

    kalshi = KalshiClient()
    poly = PolymarketClient()

    if RUN_CROSS:
        print("  Cross-platform scan:        ENABLED (--cross)")
    if RUN_BINARY:
        print("  Binary cross-platform scan: ENABLED (--binary)")

    while True:
        try:
            await scan_once(kalshi, poly)
        except Exception as e:
            logger.error(f"Scan error: {e}", exc_info=True)

        if run_once:
            break

        print(f"\n  Sleeping {SCAN_INTERVAL}s until next scan...")
        await asyncio.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    run_once = "--once" in sys.argv
    asyncio.run(main(run_once=run_once))

