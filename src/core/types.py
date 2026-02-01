from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

class TradeType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class Action(BaseModel):
    """Represents a decision made by the agent."""
    action_type: TradeType
    market_id: str
    quantity: int = Field(default=0, ge=0)
    price: Optional[float] = None # Limit price, if applicable. None for Market.
    reasoning: str # The 'Chain of Thought' or justification
    journal: str # Key facts to remember for tomorrow
    belief: float = Field(..., ge=0.0, le=1.0) # The agent's subjective probability of YES

class MarketSnapshot(BaseModel):
    """Raw market data at a specific timestamp."""
    market_id: str
    timestamp: datetime
    best_bid: float
    best_ask: float
    last_price: float
    volume: int
    open_interest: int
    chart_data: Optional[Dict[str, Any]] = None # Could be OHLCV series
    order_book: Optional[Dict[str, Any]] = None # Deep order book if available

class NewsItem(BaseModel):
    """A piece of information available to the agent."""
    timestamp: datetime
    source: str
    headline: str
    content: str
    image_url: Optional[str] = None
    metadata: Dict[str, Any] = {}

class Observation(BaseModel):
    """The full state visible to the agent at a given step."""
    timestamp: datetime
    market_snapshots: Dict[str, MarketSnapshot]
    news: List[NewsItem]
    portfolio: 'PortfolioState'
    previous_reasoning: Optional[str] = None
    previous_journal: Optional[str] = None # From the previous day

class PortfolioState(BaseModel):
    """Current financial state of the agent."""
    cash: float
    positions: Dict[str, int] # market_id -> quantity (positive for YES, usually)
    unrealized_pnl: float
    realized_pnl: float
    total_value: float

Observation.update_forward_refs()
