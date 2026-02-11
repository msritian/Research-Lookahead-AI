import os
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from ..core.types import NewsItem
from .market import DataProvider

from exa_py import Exa
from tavily import TavilyClient

# --- Interfaces ---

class BaseContextSource(ABC):
    @abstractmethod
    def fetch(self, query: str, start_date: datetime, end_date: datetime) -> List[NewsItem]:
        pass

# --- Implementations ---

class ExaSource(BaseContextSource):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("EXA_API_KEY")
        self.exa = Exa(self.api_key) if self.api_key else None

    def fetch(self, query: str, start_date: datetime, end_date: datetime) -> List[NewsItem]:
        if not self.exa:
            return []
        
        try:
            # Format dates for Exa (ISO string)
            start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Exa search with content and images
            response = self.exa.search_and_contents(
                query,
                start_published_date=start_str,
                end_published_date=end_str,
                num_results=3,
                text=True # Ensure text is returned
            )
            
            news_items = []
            for result in response.results:
                pub_date = start_date # Fallback
                if hasattr(result, 'published_date') and result.published_date:
                    try:
                        pub_date = datetime.fromisoformat(result.published_date.replace("Z", "+00:00"))
                    except: pass

                # STRICT FILTER: Skip if result is from the future relative to simulation
                if pub_date.replace(tzinfo=None) > end_date.replace(tzinfo=None):
                    continue

                # Exa results might have an image field in metadata or as a direct attribute
                image_url = getattr(result, 'image', None)

                news_items.append(NewsItem(
                    timestamp=pub_date,
                    source="Exa",
                    headline=result.title or "No Title",
                    content=result.text[:500] if result.text else "No Content",
                    image_url=image_url
                ))
            return news_items
        except Exception as e:
            print(f"Exa Error: {e}")
            return []

class TavilySource(BaseContextSource):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.tavily = TavilyClient(api_key=self.api_key) if self.api_key else None

    def fetch(self, query: str, start_date: datetime, end_date: datetime) -> List[NewsItem]:
        if not self.tavily:
            return []
            
        try:
            # Format dates for Tavily (YYYY-MM-DD)
            # Use a slightly wider window for fetching, but end_date is the hard constraint
            tavily_start = start_date.strftime("%Y-%m-%d")
            tavily_end = end_date.strftime("%Y-%m-%d")

            # Append temporal context to the query to guide the search engine ranking
            temporal_query = f"{query} news" # Keep it simple as we have hard filters now
            
            # Tavily search with hard date filters and images
            response = self.tavily.search(
                query=temporal_query, 
                search_depth="advanced", 
                max_results=5, 
                include_images=True,
                start_date=tavily_start,
                end_date=tavily_end
            )
            
            images = response.get('images', [])
            results = response.get('results', [])
            
            news_items = []
            for i, result in enumerate(results):
                content = result.get('content', '')
                img_url = images[i] if i < len(images) else None
                
                # Check for publication date in result if available
                # Tavily sometimes provides 'published_date' in results
                item_date = start_date
                pub_date_str = result.get('published_date')
                if pub_date_str:
                    try:
                        item_date = datetime.fromisoformat(pub_date_str.split('T')[0])
                    except: pass

                news_items.append(NewsItem(
                    timestamp=item_date, 
                    source="Tavily",
                    headline=result.get('title', 'No Title'),
                    content=content,
                    image_url=img_url
                ))
            return news_items
        except Exception as e:
            print(f"Tavily Error: {e}")
            return []

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
        Fetches news from all configured sources and applies a temporal guard.
        """
        all_news = []
        query = self.query_template.format(ticker=market_context)
        
        for source in self.sources:
            try:
                items = source.fetch(query, timestamp_start, timestamp_end)
                all_news.extend(items)
            except Exception as e:
                print(f"Error fetching from source {source}: {e}")
                
        # --- Temporal Guard: Filter out 'Future Leaks' ---
        clean_news = []
        
        for item in all_news:
            content_lower = item.content.lower()
            headline_lower = item.headline.lower()
            
            # 1. Hard Year Check: If it mentions 2025 in a context that isn't future-gazing
            if timestamp_end.year <= 2024:
                if "2025" in content_lower and "inauguration" not in content_lower:
                    # Likely a recap from the future
                    print(f"DEBUG: Filtered out 2025 leak: {item.headline}")
                    continue

            # 2. Results Check (Before Nov 2024)
            if timestamp_end.year == 2024 and timestamp_end.month < 11:
                # Forbidden 'Outcome' Keywords
                outcome_patterns = [
                    "trump wins", "harris defeats", "trump defeats", "won the presidency",
                    "312 electoral", "226 electoral", "president-elect", "defeated kamala",
                    "landslide victory", "election results map", "certified the victory",
                    "formally declared him the winner"
                ]
                if any(p in content_lower or p in headline_lower for p in outcome_patterns):
                    print(f"DEBUG: Filtered out outcome leak: {item.headline}")
                    continue

            # 3. Source Sanity Check
            # If the source timestamp itself is after our end_date (sometimes sources lie in their metadata)
            if item.timestamp and item.timestamp.replace(tzinfo=None) > timestamp_end.replace(tzinfo=None):
                print(f"DEBUG: Filtered out metadata leak: {item.headline}")
                continue
            
            clean_news.append(item)

        return clean_news
