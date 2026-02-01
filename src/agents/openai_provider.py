import os
from typing import List, Optional
from openai import OpenAI
from src.core.llm_interface import LLMProvider

class OpenAIProvider(LLMProvider):
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        # Use provided key or fallback to env var
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str, image_urls: Optional[List[str]] = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        user_content = []
        user_content.append({"type": "text", "text": user_prompt})

        # Add images if available
        if image_urls:
            for url in image_urls:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                        "detail": "high"
                    }
                })
        
        messages.append({"role": "user", "content": user_content})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0, # Deterministic for eval
                response_format={"type": "json_object"} # Enforce JSON mode
            )
            return response.choices[0].message.content
        except Exception as e:
            # Propagate error to agent to handle (or fallback)
            raise e
