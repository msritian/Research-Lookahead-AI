import json
import logging
from typing import Optional
from src.core.agent import Agent
from src.core.types import Observation, Action, TradeType
from src.core.llm_interface import LLMProvider
from src.agents.prompts import get_system_prompt, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class SequentialLLMAgent(Agent):
    def __init__(self, provider: LLMProvider, market_question: str):
        self.provider = provider
        self.market_question = market_question

    def act(self, observation: Observation) -> Action:
        # 1. Format Market Data
        market_strs = []
        for mid, snap in observation.market_snapshots.items():
            market_strs.append(
                f"ID: {mid} | Price: {snap.last_price:.2f} | Bid: {snap.best_bid:.2f} | Ask: {snap.best_ask:.2f} | Vol: {snap.volume}"
            )
        market_data_str = "\n".join(market_strs)

        # 2. Format News and collect all images
        news_strs = []
        image_urls = []
        
        # 2a. News images
        for n in observation.news:
            news_strs.append(f"[{n.source}] {n.headline}: {n.content[:200]}...")
            if n.image_url:
                image_urls.append(n.image_url)
        news_str = "\n".join(news_strs) if news_strs else "No new news today."

        # 2b. Market chart images
        for mid, snap in observation.market_snapshots.items():
            if snap.image_url:
                image_urls.append(snap.image_url)

        # 3. Format Portfolio
        positions_str = str(observation.portfolio.positions)
        
        # 4. Construct Prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            date=observation.timestamp.strftime("%Y-%m-%d"),
            cash=observation.portfolio.cash,
            positions=positions_str,
            market_data_str=market_data_str,
            news_str=news_str,
            previous_reasoning=observation.previous_reasoning or "None (Day 1)",
            previous_journal=observation.previous_journal or "None",
            market_question=self.market_question
        )

        try:
            # 5. Call LLM
            system_prompt = get_system_prompt(self.market_question)
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
                journal=data["journal"],
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
                journal="Error occurred. No state preserved.",
                belief=0.5
            )
