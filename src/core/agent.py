from abc import ABC, abstractmethod
from typing import List
import random
from .types import Observation, Action, TradeType

class Agent(ABC):
    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """
        Decide on an action given the current observation.
        """
        pass

class RandomAgent(Agent):
    """
    An agent that takes random actions. Useful for baseline and testing.
    """
    def act(self, observation: Observation) -> Action:
        # Pick a random market from the observation
        if not observation.market_snapshots:
            # Should not happen in this setup, but handle gracefully
            return Action(
                action_type=TradeType.HOLD,
                market_id="none",
                reasoning="No markets available.",
                belief=0.5
            )

        market_id = list(observation.market_snapshots.keys())[0] # Just pick the first one for simplicity
        
        action_type = random.choice(list(TradeType))
        quantity = 0
        if action_type != TradeType.HOLD:
            quantity = random.randint(1, 10)
        
        return Action(
            action_type=action_type,
            market_id=market_id,
            quantity=quantity,
            reasoning="Random choice for testing purposes.",
            belief=random.random()
        )
