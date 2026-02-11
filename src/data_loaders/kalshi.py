import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from ..core.types import MarketSnapshot, NewsItem
from .market import DataProvider

class KalshiDataProvider(DataProvider):
    BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key # Most public data doesn't need it
        self.headers = {"Content-Type": "application/json"}
        # Cache for history to avoid repeated calls
        self._history_cache = {} 

    def get_market_snapshot(self, market_id: str, timestamp: datetime) -> MarketSnapshot:
        """
        In a real historical replay, asking the API for 'snapshot at time T' is hard because 
        /orderbook gives *current* state.
        
        Strategy:
        1. Fetch *trades* history up to timestamp.
        2. Use the last trade price as 'last_price'.
        3. For Bid/Ask, we might have to mock it based on last price (e.g. spread) 
           UNLESS we have a recorded tape of orderbooks.
        """
        
        # 1. Fetch History (if not cached)
        if market_id not in self._history_cache:
            self._fetch_history(market_id)
        
        # 2. Filter history <= timestamp
        history = self._history_cache.get(market_id, [])
        relevant_trades = [t for t in history if t['ts'] <= timestamp.timestamp()]
        
        if not relevant_trades:
            # No data yet? Return placeholder
            return MarketSnapshot(
                market_id=market_id,
                timestamp=timestamp,
                best_bid=0.01,
                best_ask=0.99,
                last_price=0.50,
                volume=0,
                open_interest=0
            )

        last_trade = relevant_trades[-1]
        price = last_trade['price'] / 100.0 # Kalshi provides cents usually

        # Mocking Bid/Ask because historical orderbook is not available via simple API
        # In a real tape production system, we would load this from a DB.
        return MarketSnapshot(
            market_id=market_id,
            timestamp=timestamp,
            best_bid=max(0.01, price - 0.02),
            best_ask=min(0.99, price + 0.02),
            last_price=price,
            volume=len(relevant_trades), # Simplified volume
            open_interest=0 # Not easily available historically
        )

    def get_news(self, timestamp_start: datetime, timestamp_end: datetime) -> List[NewsItem]:
        # Web Search implementation would go here
        # For now, return empty list or mock
        return []

    def _fetch_history(self, market_id: str):
        # GET /markets/{ticker}/trades
        # This is strictly a demo wrapper. Real implementation needs pagination for long histories.
        url = f"{self.BASE_URL}/markets/{market_id}/trades"
        try:
            params = {"limit": 1000} # Get a good chunk of recent trades
            resp = requests.get(url, headers=self.headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            if 'trades' in data:
                # Sort by timestamp ascending for simulation playback
                self._history_cache[market_id] = sorted(data['trades'], key=lambda x: x['ts'])
                print(f"Fetched {len(self._history_cache[market_id])} trades for {market_id}")
            else:
                print(f"No trades found for {market_id}")
                self._history_cache[market_id] = []
                
        except Exception as e:
            print(f"Error fetching Kalshi history for {market_id}: {e}")
            self._history_cache[market_id] = []
