# Arbitrage logic has been consolidated into:
#   kalshi/kalshi_client.py  → KalshiClient.scan_mece_opportunities()
#   polymarket/polymarket_client.py → PolymarketClient.get_neg_risk_opportunities()
#   main.py → scan loop + execution
#
# This file kept for import compatibility.

from kalshi.kalshi_client import KalshiClient
from polymarket.polymarket_client import PolymarketClient

__all__ = ["KalshiClient", "PolymarketClient"]
