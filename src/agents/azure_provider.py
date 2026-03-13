import os
import base64
import mimetypes
import requests
import io
from typing import List, Optional
from PIL import Image
from openai import AzureOpenAI
from src.core.llm_interface import LLMProvider

class AzureOpenAIProvider(LLMProvider):
    def __init__(self, deployment: str = "gpt-4o", api_key=None, endpoint=None, api_version=None):
        self.deployment = deployment
        self.client = AzureOpenAI(
            api_key=api_key or os.environ.get("AZURE_OPENAI_KEY"),
            azure_endpoint=endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "https://api-backup.openai.azure.com/"),
            api_version=api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        )

    def _encode_image(self, image_source: str) -> Optional[str]:
        """
        Downloads and encodes a local path or remote URL into a base64 data URL.
        Returns None if the image cannot be fetched or is not a valid image type.
        """
        try:
            if os.path.exists(image_source):
                mime_type, _ = mimetypes.guess_type(image_source)
                with open(image_source, "rb") as f:
                    data = f.read()
            else:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
                }
                resp = requests.get(image_source, headers=headers, timeout=8)
                resp.raise_for_status()
                mime_type = resp.headers.get('Content-Type', '').split(';')[0].strip()
                data = resp.content

            ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/gif', 'image/webp']

            if mime_type == 'image/avif':
                try:
                    print(f"[Image Conv] Converting AVIF to JPEG: {image_source[:60]}...")
                    img = Image.open(io.BytesIO(data))
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    output = io.BytesIO()
                    img.save(output, format="JPEG", quality=85)
                    data = output.getvalue()
                    mime_type = "image/jpeg"
                except Exception as conv_err:
                    print(f"[Image Skip] Failed to convert AVIF: {conv_err}")
                    return None

            if not mime_type or mime_type not in ALLOWED_TYPES:
                if mime_type:
                    print(f"[Image Skip] {image_source[:60]}... → Unsupported format: {mime_type}")
                return None

            if len(data) < 1000:
                print(f"[Image Skip] {image_source[:60]}... → File too small ({len(data)} bytes)")
                return None

            encoded = base64.b64encode(data).decode('utf-8')
            return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            print(f"[Image Skip] {image_source[:80]}... → {type(e).__name__}: {e}")
            return None

    def generate(self, system_prompt: str, user_prompt: str, image_urls: Optional[List[str]] = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        user_content = []
        user_content.append({"type": "text", "text": user_prompt})

        if image_urls:
            encoded_count = 0
            for url in image_urls:
                if not url or encoded_count >= 3:
                    break
                image_payload = self._encode_image(url)
                if image_payload:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_payload,
                            "detail": "low"
                        }
                    })
                    encoded_count += 1
            if encoded_count > 0:
                print(f"[Vision] Sending {encoded_count} image(s) to Azure OpenAI.")

        messages.append({"role": "user", "content": user_content})

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            raise e
