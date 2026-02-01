from src.core.llm_interface import LLMProvider
from typing import List, Optional
import json

class MockLLMProvider(LLMProvider):
    def generate(self, system_prompt: str, user_prompt: str, image_urls: Optional[List[str]] = None) -> str:
        # Simulate a generic response
        # We can look at the user_prompt to possibly switch behavior, but for now just return a valid BUY action
        response = {
            "action": "BUY",
            "market_id": "kalshi_mut_1",
            "quantity": 1,
            "belief_probability": 0.75,
            "reasoning": "Positive fake news detected. Increasing position.",
            "journal": "Day 1: Market seems bullish."
        }
        return json.dumps(response)
