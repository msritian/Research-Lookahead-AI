import os
import base64
import mimetypes
import requests
from typing import List, Optional
from openai import OpenAI
from src.core.llm_interface import LLMProvider

class OpenAIProvider(LLMProvider):
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        # Use provided key or fallback to env var
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name

    def _encode_image(self, image_source: str) -> Optional[str]:
        """
        Encodes a local path or a remote URL into a base64 data URL.
        """
        try:
            if os.path.exists(image_source):
                # Local file
                mime_type, _ = mimetypes.guess_type(image_source)
                with open(image_source, "rb") as f:
                    data = f.read()
            else:
                # Remote URL
                resp = requests.get(image_source, timeout=5)
                resp.raise_for_status()
                data = resp.content
                mime_type = resp.headers.get('Content-Type', 'image/png')
            
            encoded = base64.b64encode(data).decode('utf-8')
            return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            print(f"Error encoding image {image_source}: {e}")
            return None

    def generate(self, system_prompt: str, user_prompt: str, image_urls: Optional[List[str]] = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        user_content = []
        user_content.append({"type": "text", "text": user_prompt})

        # Add images if available
        if image_urls:
            for url in image_urls:
                if not url: continue
                b64_url = self._encode_image(url)
                if b64_url:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": b64_url,
                            "detail": "high"
                        }
                    })
        
        messages.append({"role": "user", "content": user_content})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            raise e
