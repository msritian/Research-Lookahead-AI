from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
import json
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
        market_ids: List[str] = None,
        context_window_days: int = 14,
        run_dir: str = "runs/default"
    ):
        self.current_time = start_date
        self.end_date = end_date
        self.market_provider = market_provider
        self.context_provider = context_provider
        self.agent = agent
        self.logger = logger
        self.portfolio = Portfolio()
        self.step_size = step_size
        self.market_ids = market_ids or ["market_1"]
        self.context_window_days = context_window_days
        
        self.history: List[Dict[str, Any]] = []
        self.run_dir = run_dir
        self.raw_data_dir = os.path.join(run_dir, "raw_data")
        self.charts_dir = os.path.join(run_dir, "charts")
        
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # Propagate chart dir to provider
        if hasattr(self.market_provider, 'charts_dir'):
            self.market_provider.charts_dir = self.charts_dir

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

        # News/Context from the T-window days until just before current_time
        # Cutoff is (current_time - 1s) so any news published ON day T is excluded.
        # The market snapshot price is taken at current_time (midnight), so news from
        # the same day could reveal post-market info — we strictly avoid this.
        news_start = self.current_time - timedelta(days=self.context_window_days)
        news_end = self.current_time - timedelta(seconds=1)
        news = []
        market_context = self.market_ids[0] if self.market_ids else "General"
        if self.context_provider:
             news = self.context_provider.get_news(news_start, news_end, market_context=market_context)
        elif self.market_provider:
             news = self.market_provider.get_news(news_start, news_end)

        # 3. Construct Observation
        observation = Observation(
            timestamp=self.current_time,
            context_window_days=self.context_window_days,
            market_snapshots=snapshots,
            news=news,
            portfolio=self.portfolio.get_state(current_prices)
        )

        # Retrieve ground truth rules to pass to the agent
        market_rules = "None Provided"
        if hasattr(self.market_provider, 'get_market_rules'):
            market_rules = self.market_provider.get_market_rules(self.market_ids[0])

        # 4. Agent Action
        action = self.agent.act(observation, market_rules=market_rules)

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
            "execution_price": execution_price if action.action_type != TradeType.HOLD else None,
            "portfolio_value": self.portfolio.get_state(current_prices).total_value,
            "action": action.dict(),
            "observation": {
                "news": [n.dict() for n in news],
                "portfolio": observation.portfolio.dict()
            },
            "ground_truth_verification": {
                # Store the direct mapping of actual price vs. agent's belief probability to calculate Brier Score later
                "actual_prices": current_prices, 
                "agent_belief": action.belief
            },
            "success": success
        }
        self.history.append(log_entry)
        
        if self.logger:
            self.logger.log("step", log_entry)

        # 7. Save raw data for human inspection
        raw_step = {
            "date": self.current_time.strftime("%Y-%m-%d"),
            "context_window": f"{news_start.date()} to {news_end.date()} (cutoff: {news_end.strftime('%Y-%m-%d %H:%M:%S')})",
            "market_data": {
                mid: {
                    "price": snap.last_price,
                    "bid": snap.best_bid,
                    "ask": snap.best_ask,
                    "volume": snap.volume,
                    "chart_image": snap.image_url
                }
                for mid, snap in snapshots.items()
            },
            "market_rules": market_rules,
            "news": [
                {
                    "date": n.timestamp.strftime("%Y-%m-%d"),
                    "source": n.source,
                    "headline": n.headline,
                    "content": n.content,
                    "image_url": n.image_url
                }
                for n in news
            ],
            "agent_action": {
                "action": action.action_type,
                "belief": action.belief,
                "reasoning": action.reasoning
            }
        }
        raw_path = os.path.join(self.raw_data_dir, f"{self.current_time.strftime('%Y-%m-%d')}.json")
        with open(raw_path, 'w') as f:
            json.dump(raw_step, f, indent=2, default=str)

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
