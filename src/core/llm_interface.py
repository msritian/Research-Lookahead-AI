from abc import ABC, abstractmethod
from typing import List, Optional

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, image_urls: Optional[List[str]] = None) -> str:
        """
        Generates a response from the LLM.
        
        Args:
            system_prompt: High-level instructions (Role, Output format).
            user_prompt: The specific context for this turn.
            image_urls: List of URLs for multimodal input.
            
        Returns:
            The raw text response.
        """
        pass
