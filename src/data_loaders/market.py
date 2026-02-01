from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict
from ..core.types import MarketSnapshot, NewsItem
import random

class DataProvider(ABC):
    @abstractmethod
    def get_market_snapshot(self, market_id: str, timestamp: datetime) -> MarketSnapshot:
        pass

    @abstractmethod
    def get_news(self, timestamp_start: datetime, timestamp_end: datetime) -> List[NewsItem]:
        pass

class MockDataProvider(DataProvider):
    """
    Generates random walk data for testing purposes.
    """
    def __init__(self):
        self._prices: Dict[str, float] = {}

    def get_market_snapshot(self, market_id: str, timestamp: datetime) -> MarketSnapshot:
        # Simple random walk for price
        current_price = self._prices.get(market_id, 0.50)
        change = random.uniform(-0.05, 0.05)
        new_price = max(0.01, min(0.99, current_price + change))
        self._prices[market_id] = new_price

        return MarketSnapshot(
            market_id=market_id,
            timestamp=timestamp,
            best_bid=new_price - 0.01,
            best_ask=new_price + 0.01,
            last_price=new_price,
            volume=random.randint(100, 10000),
            open_interest=random.randint(1000, 50000),
            chart_data={"type": "mock_line", "val": new_price}
        )

    def get_news(self, timestamp_start: datetime, timestamp_end: datetime) -> List[NewsItem]:
        # Occasionally generate a fake news item
        if random.random() < 0.3:
            return [NewsItem(
                timestamp=timestamp_start,
                source="MockNews",
                headline=f"Random Event occurred at {timestamp_start}",
                content="This is a generated news item for testing.",
                image_url="http://placeholder.com/chart.png"
            )]
        return []
