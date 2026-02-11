import requests
import json
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Optional
from ..core.types import MarketSnapshot, NewsItem
from .market import DataProvider

class PolymarketDataProvider(DataProvider):
    GAMMA_URL = "https://gamma-api.polymarket.com"
    CLOB_URL = "https://clob.polymarket.com"

    def __init__(self):
        self._token_cache: Dict[str, str] = {} # ticker -> clobTokenId
        self._history_cache: Dict[str, List[Dict[str, Any]]] = {} # token_id -> history
        self._fetched_ranges: Dict[str, List[tuple]] = {} # token_id -> [(start, end)]
        self.charts_dir = "charts"
        os.makedirs(self.charts_dir, exist_ok=True)

    def get_market_snapshot(self, market_id: str, timestamp: datetime) -> MarketSnapshot:
        # 1. Resolve to Token ID
        token_id = self._resolve_token(market_id)
        if not token_id:
            return self._placeholder(market_id, timestamp)

        # 2. Fetch History for the relevant range if needed
        ts_val = timestamp.timestamp()
        if not self._is_range_covered(token_id, ts_val):
            self._fetch_history_window(token_id, ts_val)

        # 3. Get cached history
        history = self._history_cache.get(token_id, [])
        relevant = [p for p in history if p['t'] <= ts_val]

        if not relevant:
            return self._placeholder(market_id, timestamp)

        last_p = relevant[-1]
        # Valid if within 2 days
        if abs(ts_val - last_p['t']) > 86400 * 2:
             return self._placeholder(market_id, timestamp)

        price = float(last_p['p'])
        
        # 4. Generate Chart Image
        chart_path = self._generate_chart_image(token_id, market_id, relevant, timestamp)

        return MarketSnapshot(
            market_id=market_id,
            timestamp=timestamp,
            best_bid=max(0.01, price - 0.005),
            best_ask=min(0.99, price + 0.005),
            last_price=price,
            volume=0,
            open_interest=0,
            image_url=chart_path
        )

    def _generate_chart_image(self, token_id: str, market_id: str, history: List[Dict[str, Any]], current_ts: datetime) -> Optional[str]:
        """
        Generates a price chart for the given history.
        """
        if not history: return None
        
        try:
            # Last 7 days of history for the chart
            window_start = current_ts.timestamp() - (86400 * 7)
            chart_data = [p for p in history if p['t'] >= window_start]
            
            if len(chart_data) < 2: return None
            
            times = [datetime.fromtimestamp(p['t']) for p in chart_data]
            prices = [float(p['p']) for p in chart_data]
            
            plt.figure(figsize=(10, 5))
            plt.plot(times, prices, marker=None, linestyle='-', color='#007aff')
            plt.title(f"Price History: {market_id}")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            
            # Formatting
            plt.gcf().autofmt_xdate()
            
            filename = f"{token_id}_{int(current_ts.timestamp())}.png"
            filepath = os.path.join(self.charts_dir, filename)
            plt.savefig(filepath)
            plt.close()
            
            return os.path.abspath(filepath)
        except Exception as e:
            print(f"Error generating chart: {e}")
            return None

    def get_news(self, timestamp_start: datetime, timestamp_end: datetime) -> List[NewsItem]:
        return []

    def _is_range_covered(self, token_id: str, ts: float) -> bool:
        ranges = self._fetched_ranges.get(token_id, [])
        for start, end in ranges:
            if start <= ts <= end:
                return True
        return False

    def _resolve_token(self, query: str) -> Optional[str]:
        if query in self._token_cache:
            return self._token_cache[query]

        print(f"Searching Polymarket for: {query}")
        try:
            url = f"{self.GAMMA_URL}/public-search"
            resp = requests.get(url, params={"q": query, "active": "true"})
            resp.raise_for_status()
            search_data = resp.json()

            markets = []
            if isinstance(search_data, dict) and "events" in search_data:
                for event in search_data["events"]:
                    markets.extend(event.get("markets", []))
            elif isinstance(search_data, list):
                markets = search_data
            
            if not markets:
                url = f"{self.GAMMA_URL}/markets"
                resp = requests.get(url, params={"active": "true", "limit": 20})
                resp.raise_for_status()
                markets = resp.json()

            def score_market(m):
                q_text = m.get("question", "").lower()
                query_tokens = query.lower().split()
                matches = sum(1 for token in query_tokens if token in q_text)
                volume = float(m.get("volume", 0) or 0)
                boost = 1000 if all(t in q_text for t in query_tokens) else 0
                return matches * 100 + boost + (volume / 1000000)

            markets.sort(key=score_market, reverse=True)

            for market in markets:
                if not isinstance(market, dict): continue
                tokens_raw = market.get("clobTokenIds", "[]")
                outcomes_raw = market.get("outcomes", "[]")
                try:
                    token_ids = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
                    outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
                except: continue
                
                if outcomes and token_ids and len(outcomes) == len(token_ids):
                    yes_idx = -1
                    for i, o in enumerate(outcomes):
                        if o.lower() == "yes":
                            yes_idx = i
                            break
                    if yes_idx != -1:
                        tid = token_ids[yes_idx]
                        self._token_cache[query] = tid
                        print(f"Resolved {query} to token: {tid}")
                        return tid
            return None
        except Exception as e:
            print(f"Error resolving Polymarket token: {e}")
            return None

    def _fetch_history_window(self, token_id: str, center_ts: float):
        url = f"{self.CLOB_URL}/prices-history"
        # Fetch 14 days around center_ts (verified range limit)
        start_ts = int(center_ts - 86400 * 7)
        end_ts = int(center_ts + 86400 * 7)
        
        now = int(time.time())
        if end_ts > now: end_ts = now

        try:
            print(f"Fetching history window for {token_id}: {datetime.fromtimestamp(start_ts)} to {datetime.fromtimestamp(end_ts)}")
            resp = requests.get(url, params={
                "market": token_id, 
                "startTs": start_ts, 
                "endTs": end_ts,
                "fidelity": 60 
            })
            resp.raise_for_status()
            data = resp.json()
            
            new_history = data.get("history", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            
            if new_history:
                print(f"Received {len(new_history)} price points")
                formatted = []
                for entry in new_history:
                    if isinstance(entry, dict) and 't' in entry and 'p' in entry:
                        formatted.append(entry)
                    elif isinstance(entry, list) and len(entry) >= 2:
                        formatted.append({'t': entry[0], 'p': entry[1]})
                
                existing = self._history_cache.get(token_id, [])
                all_history = existing + formatted
                unique_history = {h['t']: h['p'] for h in all_history}
                self._history_cache[token_id] = sorted([{'t': t, 'p': p} for t, p in unique_history.items()], key=lambda x: x['t'])
                
                if token_id not in self._fetched_ranges: self._fetched_ranges[token_id] = []
                self._fetched_ranges[token_id].append((start_ts, end_ts))
            else:
                print(f"No history in window for {token_id}")
                if token_id not in self._fetched_ranges: self._fetched_ranges[token_id] = []
                self._fetched_ranges[token_id].append((start_ts, end_ts))

        except Exception as e:
            print(f"Error fetching Polymarket history: {e}")

    def _placeholder(self, market_id: str, timestamp: datetime) -> MarketSnapshot:
        return MarketSnapshot(
            market_id=market_id,
            timestamp=timestamp,
            best_bid=0.49,
            best_ask=0.51,
            last_price=0.50,
            volume=0,
            open_interest=0
        )
