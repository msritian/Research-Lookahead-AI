from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from .types import Observation, Action, TradeType, MarketSnapshot, PortfolioState
from .portfolio import Portfolio
from .agent import Agent
from ..data_loaders.market import DataProvider
from ..utils.logger import ExperimentLogger

class MarketEnvironment:
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        market_provider: DataProvider,
        agent: Agent,
        context_provider: Optional[DataProvider] = None,
        logger: Optional[ExperimentLogger] = None,
        step_size: timedelta = timedelta(days=1),
        market_ids: List[str] = None
    ):
        self.current_time = start_date
        self.end_date = end_date
        self.market_provider = market_provider
        self.context_provider = context_provider
        self.agent = agent
        self.logger = logger
        self.portfolio = Portfolio()
        self.step_size = step_size
        self.market_ids = market_ids or ["market_1"] # Default to one market if not specified
        
        self.history: List[Dict[str, Any]] = []
        self.previous_reasoning: Optional[str] = None
        self.previous_journal: Optional[str] = None

    def step(self):
        """
        Advances the simulation by one step.
        """
        if self.current_time >= self.end_date:
            return False # Simulation finished

        # 1. Morning State Capture
        snapshots = {}
        current_prices = {}
        for m_id in self.market_ids:
            snapshot = self.market_provider.get_market_snapshot(m_id, self.current_time)
            snapshots[m_id] = snapshot
            current_prices[m_id] = snapshot.last_price

        # News/Context from the last step until now
        news_start = self.current_time - self.step_size
        news = []
        market_context = self.market_ids[0] if self.market_ids else "General"
        if self.context_provider:
             news = self.context_provider.get_news(news_start, self.current_time, market_context=market_context)
        elif self.market_provider:
             news = self.market_provider.get_news(news_start, self.current_time)

        # 3. Construct Observation
        observation = Observation(
            timestamp=self.current_time,
            market_snapshots=snapshots,
            news=news,
            portfolio=self.portfolio.get_state(current_prices),
            previous_reasoning=self.previous_reasoning,
            previous_journal=self.previous_journal
        )

        # 4. Agent Action
        action = self.agent.act(observation)
        self.previous_reasoning = action.reasoning 
        self.previous_journal = action.journal

        # 5. Execution
        # Get the price for the trade
        execution_price = 0.0
        if action.market_id in snapshots:
            # Simple assumption: Buy at Ask, Sell at Bid
            # In a real order book model, we would match against the book.
            snapshot = snapshots[action.market_id]
            if action.action_type == TradeType.BUY:
                execution_price = snapshot.best_ask
            elif action.action_type == TradeType.SELL:
                execution_price = snapshot.best_bid
        
        success = self.portfolio.execute_trade(
            action.market_id,
            action.action_type,
            action.quantity,
            execution_price
        )

        # 6. Logging
        log_entry = {
            "timestamp": self.current_time.isoformat(),
            "market_prices": current_prices,
            "portfolio_value": self.portfolio.get_state(current_prices).total_value,
            "action": action.dict(),
            "observation": {
                "news": [n.dict() for n in news],
                "portfolio": observation.portfolio.dict()
            },
            "success": success
        }
        self.history.append(log_entry)
        
        if self.logger:
            self.logger.log("step", log_entry)

        # 7. Time Advance
        self.current_time += self.step_size
        return True

    def run(self):
        """
        Runs the simulation until the end date.
        """
        print(f"Starting simulation from {self.current_time} to {self.end_date}")
        while self.step():
            pass
        print("Simulation complete.")
