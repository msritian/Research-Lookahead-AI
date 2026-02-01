import os
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from ..core.types import NewsItem
from .market import DataProvider

# --- Interfaces ---

class BaseContextSource(ABC):
    @abstractmethod
    def fetch(self, query: str, start_date: datetime, end_date: datetime) -> List[NewsItem]:
        pass

# --- Implementations ---

class ExaSource(BaseContextSource):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("EXA_API_KEY")
        # Initialize client here if library is available
        # from exa_py import Exa
        # self.exa = Exa(self.api_key) if self.api_key else None

    def fetch(self, query: str, start_date: datetime, end_date: datetime) -> List[NewsItem]:
        if not self.api_key:
            return []
        
        # Mock implementation until library is confirmed installed
        # Real impl would involve self.exa.search_and_contents(...)
        return [
            NewsItem(
                timestamp=start_date,
                source="Exa",
                headline=f"Exa Result for {query}",
                content="Exa content placeholder...",
                image_url=None
            )
        ]

class TavilySource(BaseContextSource):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")

    def fetch(self, query: str, start_date: datetime, end_date: datetime) -> List[NewsItem]:
        if not self.api_key:
            return []
            
        # Mock implementation
        return [
            NewsItem(
                timestamp=start_date,
                source="Tavily",
                headline=f"Tavily Result for {query}",
                content="Tavily content placeholder...",
                image_url=None
            )
        ]

class WebSource(BaseContextSource):
    def fetch(self, query: str, start_date: datetime, end_date: datetime) -> List[NewsItem]:
        # Fallback using requests or standard search
        return []

# --- Aggregator ---

class ContextDataProvider(DataProvider):
    def __init__(self, sources: List[str] = ["exa", "tavily"], query_template: str = "{ticker} context"):
        self.sources = []
        if "exa" in sources:
            self.sources.append(ExaSource())
        if "tavily" in sources:
            self.sources.append(TavilySource())
        if "web" in sources:
            self.sources.append(WebSource())
            
        self.query_template = query_template

    def get_market_snapshot(self, market_id: str, timestamp: datetime):
        raise NotImplementedError("This provider only handles News")

    def get_news(self, timestamp_start: datetime, timestamp_end: datetime, market_context: str = "General") -> List[NewsItem]:
        """
        Fetches news from all configured sources.
        """
        all_news = []
        query = self.query_template.format(ticker=market_context)
        
        for source in self.sources:
            try:
                items = source.fetch(query, timestamp_start, timestamp_end)
                all_news.extend(items)
            except Exception as e:
                print(f"Error fetching from source {source}: {e}")
                
        # Deduplication could go here
        return all_news
