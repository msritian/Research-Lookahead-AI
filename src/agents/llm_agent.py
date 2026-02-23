import json
import logging
from typing import Optional
from src.core.agent import Agent
from src.core.types import Observation, Action, TradeType
from src.core.llm_interface import LLMProvider
from src.agents.prompts import get_system_prompt, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class SequentialLLMAgent(Agent):
    def __init__(self, provider: LLMProvider, market_question: str, max_content: int = 2000):
        self.provider = provider
        self.market_question = market_question
        self.max_content = max_content

    def act(self, observation: Observation, market_rules: str = "None Provided") -> Action:
        # 1. Format Market Data
        market_strs = []
        for mid, snap in observation.market_snapshots.items():
            market_strs.append(
                f"ID: {mid} | Price: {snap.last_price:.2f} | Bid: {snap.best_bid:.2f} | Ask: {snap.best_ask:.2f} | Vol: {snap.volume}"
            )
        market_data_str = "\n".join(market_strs)

        # 2. Format News and collect all images
        # 2a. Group news by Date for the 14-day timeline
        news_by_date = {}
        image_urls: list = []  # News images collected for multimodal context
        
        for n in observation.news:
            date_str = n.timestamp.strftime("%Y-%m-%d")
            if date_str not in news_by_date:
                news_by_date[date_str] = []
            news_by_date[date_str].append(f"[{n.source}] {n.headline}: {n.content[:self.max_content]}")
            if n.image_url:
                image_urls.append(n.image_url)  # Collect for multimodal context
                
        # Format the grouped string
        news_strs = []
        for d in sorted(news_by_date.keys()):
            news_strs.append(f"--- news from {d} ---")
            news_strs.extend(news_by_date[d])
            news_strs.append("") # spacer

        news_str = "\n".join(news_strs) if news_strs else "No news available for the given timeframe."

        # 2b. Polymarket chart images (local .png files — safe to base64 encode)
        for mid, snap in observation.market_snapshots.items():
            if snap.image_url:
                image_urls.append(snap.image_url)

        # 3. Format Portfolio
        positions_str = str(observation.portfolio.positions)
        
        # 4. Construct Prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            date=observation.timestamp.strftime("%Y-%m-%d"),
            window_days=observation.context_window_days,
            cash=observation.portfolio.cash,
            positions=positions_str,
            market_data_str=market_data_str,
            news_str=news_str,
            market_question=self.market_question
        )

        try:
            # 5. Call LLM
            system_prompt = get_system_prompt(self.market_question, market_rules).replace("{{window_days}}", str(observation.context_window_days))
            response_text = self.provider.generate(system_prompt, user_prompt, image_urls)
            
            # 6. Parse JSON
            # Basic cleanup to handle markdown fences if the model adds them
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
            
            return Action(
                action_type=TradeType(data["action"]),
                market_id=data["market_id"],
                quantity=int(data["quantity"]),
                reasoning=data["reasoning"],
                belief=float(data["belief_probability"])
            )
            
        except Exception as e:
            logger.error(f"Failed to generate/parse action: {e}")
            # Fallback action
            return Action(
                action_type=TradeType.HOLD,
                market_id="error_fallback",
                quantity=0,
                reasoning=f"Error in LLM processing: {str(e)}",
                belief=0.5
            )
