import os
import io
import requests
from typing import List, Optional
from PIL import Image
from src.core.llm_interface import LLMProvider

class QwenProvider(LLMProvider):
    MODEL_ID = "Qwen/Qwen2.5-Omni-3B"

    def __init__(self, device: str = "auto"):
        self.device = device
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return
        print(f"[Qwen] Loading model {self.MODEL_ID}...")
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        self._processor = Qwen2_5OmniProcessor.from_pretrained(self.MODEL_ID)
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            dtype="auto",
            device_map=self.device
        )
        print("[Qwen] Model loaded.")

    def _load_image(self, image_source: str) -> Optional[Image.Image]:
        try:
            if os.path.exists(image_source):
                img = Image.open(image_source)
            else:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
                }
                resp = requests.get(image_source, headers=headers, timeout=8)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content))

            # Convert AVIF or other formats to RGB JPEG-compatible
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            return img
        except Exception as e:
            print(f"[Image Skip] {image_source[:80]}... → {type(e).__name__}: {e}")
            return None

    def generate(self, system_prompt: str, user_prompt: str, image_urls: Optional[List[str]] = None) -> str:
        self._load_model()

        images = []
        if image_urls:
            for url in image_urls:
                if not url or len(images) >= 3:
                    break
                img = self._load_image(url)
                if img is not None:
                    images.append(img)
            if images:
                print(f"[Vision] Sending {len(images)} image(s) to Qwen.")

        # Build user content
        user_content = []
        for img in images:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt"
        ).to(self._model.device)

        import torch
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                return_audio=False,
                max_new_tokens=512,
                do_sample=False,
            )

        # Strip prompt tokens from output
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
