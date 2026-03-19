"""
QwenProvider
============
Local HuggingFace provider for Qwen2.5-3B-Instruct.

Beyond the standard LLMProvider.generate() interface, this class exposes
predict_market() which:
  1. Fetches relevant news via Exa up to a cutoff date
  2. Constructs a structured prompt with market context + price history
  3. Runs the model and extracts log-probabilities over "Yes" / "No" tokens
  4. Returns a calibrated probability and a binary prediction

Designed for Colab A100 (40GB). Model loads in BF16 (~7GB).
"""

from __future__ import annotations

import os
import math
import json
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.core.llm_interface import LLMProvider

# HuggingFace model ID
QWEN_INSTRUCT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Candidate answer tokens — we score log-prob of the first token of each
YES_TOKENS = ["Yes", "YES", "yes"]
NO_TOKENS  = ["No",  "NO",  "no"]

SYSTEM_PROMPT = """You are an expert prediction market analyst. Given a market question, relevant news, and price history, your task is to forecast whether the outcome will resolve YES or NO.

Rules:
- Reason carefully using the news and price context provided.
- Your final answer MUST be exactly "Yes" or "No" on its own line, preceded by "Answer:".
- Before your answer, briefly explain your reasoning in 2-4 sentences.
- Do not add any text after "Answer: Yes" or "Answer: No".
"""

USER_PROMPT_TEMPLATE = """## Market Question
{question}

## Resolution Context
- Market resolves YES for: {answer_yes}
- Market resolves NO for:  {answer_no}
- Cutoff date (predict as of this date): {cutoff_date}
- Price at cutoff (market-implied probability): {price_at_cutoff}

## Price History (YES token price over time, 0-1 scale)
{price_history}

## Relevant News (published before {cutoff_date})
{news_context}

## Your Analysis
Based on the above, will this market resolve YES or NO?
Provide 2-4 sentences of reasoning, then on a new line write:
Answer: Yes
or
Answer: No"""


class QwenProvider(LLMProvider):
    """
    Local Qwen2.5-3B-Instruct provider with log-prob calibration.
    """

    def __init__(
        self,
        model_name: str = QWEN_INSTRUCT_MODEL,
        exa_api_key: Optional[str] = None,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
    ):
        self.model_name  = model_name
        self.exa_api_key = exa_api_key or os.environ.get("EXA_API_KEY", "")
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model      = None
        self._tokenizer  = None
        self.load_in_4bit = load_in_4bit

    # ------------------------------------------------------------------
    # Model lazy-loading
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return

        print(f"[QwenProvider] Loading {self.model_name} on {self.device}...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self._model.eval()
        print(f"[QwenProvider] Model loaded.")

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: Optional[List[str]] = None,
    ) -> str:
        """Standard text generation (no image support for base Qwen-3B)."""
        self._load_model()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Log-prob extraction
    # ------------------------------------------------------------------

    def _get_logprob_for_token(self, prompt_ids: torch.Tensor, token_str: str) -> float:
        """
        Returns the log-probability of `token_str` being the next token
        after `prompt_ids`, averaged over its constituent sub-tokens.
        """
        token_ids = self._tokenizer.encode(
            token_str, add_special_tokens=False
        )
        if not token_ids:
            return float("-inf")

        log_prob_sum = 0.0
        current_ids = prompt_ids.clone()

        with torch.no_grad():
            for tid in token_ids:
                outputs = self._model(input_ids=current_ids)
                logits  = outputs.logits[0, -1, :]           # last position
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                log_prob_sum += log_prob_sum + log_probs[tid].item()
                # append for multi-token strings
                current_ids = torch.cat(
                    [current_ids, torch.tensor([[tid]], device=self.device)], dim=1
                )

        return log_prob_sum / len(token_ids)

    def _extract_yes_no_logprobs(self, prompt_text: str) -> Dict[str, float]:
        """
        Returns {'yes_logprob': float, 'no_logprob': float, 'yes_prob': float}.
        Aggregates over multiple surface forms (Yes/YES/yes, No/NO/no).
        """
        self._load_model()

        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self.device)
        prompt_ids = inputs["input_ids"]

        # Score all surface forms, take max log-prob per side
        yes_lp = max(
            self._get_logprob_for_token(prompt_ids, tok) for tok in YES_TOKENS
        )
        no_lp = max(
            self._get_logprob_for_token(prompt_ids, tok) for tok in NO_TOKENS
        )

        # Normalise to a probability over {Yes, No}
        yes_prob_raw = math.exp(yes_lp)
        no_prob_raw  = math.exp(no_lp)
        total = yes_prob_raw + no_prob_raw + 1e-12

        return {
            "yes_logprob": round(yes_lp, 6),
            "no_logprob":  round(no_lp, 6),
            "yes_prob":    round(yes_prob_raw / total, 6),
        }

    # ------------------------------------------------------------------
    # Exa news fetch
    # ------------------------------------------------------------------

    def _fetch_exa_news(
        self,
        query: str,
        cutoff_date: datetime,
        window_days: int = 30,
        max_results: int = 7,
        max_content: int = 1500,
    ) -> str:
        """Fetches news via Exa up to cutoff_date and returns a formatted string."""
        if not self.exa_api_key:
            return "No news available (EXA_API_KEY not set)."

        try:
            from exa_py import Exa
            exa = Exa(self.exa_api_key)

            start_date = cutoff_date - timedelta(days=window_days)
            start_str  = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_str    = cutoff_date.strftime("%Y-%m-%dT%H:%M:%SZ")

            response = exa.search_and_contents(
                query,
                start_published_date=start_str,
                end_published_date=end_str,
                num_results=max_results,
                text=True,
                highlights=True,
            )

            if not response.results:
                return "No relevant news found in the search window."

            parts = []
            for i, r in enumerate(response.results, 1):
                pub_str = getattr(r, "published_date", None)
                title   = getattr(r, "title", "No title") or "No title"
                content = getattr(r, "text", "") or ""

                # Secondary strict date filter — Exa's API date params are soft
                # and can return articles slightly outside the requested window.
                # This mirrors the guard in src/data_loaders/context.py.
                if pub_str:
                    try:
                        pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00")).replace(tzinfo=None)
                        if pub_dt > cutoff_date.replace(tzinfo=None):
                            continue   # drop future article
                        if pub_dt < (cutoff_date - timedelta(days=window_days)).replace(tzinfo=None):
                            continue   # drop stale article
                    except Exception:
                        pass  # unparseable date — keep the article

                snippet = content[:max_content].replace("\n", " ")
                parts.append(f"[{i}] {title} ({pub_str or 'unknown date'})\n{snippet}")

            if not parts:
                return "No relevant news found after date filtering."

            return "\n\n".join(parts)

        except Exception as e:
            return f"News fetch error: {e}"

    # ------------------------------------------------------------------
    # Main prediction entry point
    # ------------------------------------------------------------------

    def predict_market(
        self,
        market: Dict[str, Any],
        news_window_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Full pipeline for a single market record:
          1. Fetch Exa news up to cutoff_date
          2. Build prompt
          3. Get model generation (reasoning + Answer: Yes/No)
          4. Extract log-probs for calibrated probability
          5. Return structured result dict

        Args:
            market: One record from benchmark_markets.json
            news_window_days: How many days before cutoff to search

        Returns:
            {
              "market_id", "question", "category",
              "ground_truth",           # 1=YES, 0=NO
              "predicted_prob_yes",     # calibrated via log-probs
              "predicted_label",        # 1 if prob_yes > 0.5 else 0
              "model_text_answer",      # raw model output
              "parsed_text_label",      # label from text ("Yes"/"No")
              "yes_logprob", "no_logprob",
              "news_context_used",
            }
        """
        self._load_model()

        question      = market["question"]
        cutoff_date_s = market.get("cutoff_date") or market.get("end_date")
        cutoff_dt     = datetime.fromisoformat(cutoff_date_s).replace(tzinfo=timezone.utc)

        # --- 1. Fetch news ---
        print(f"  [Qwen] Fetching Exa news for: {question[:60]}...")
        news_text = self._fetch_exa_news(
            query=question,
            cutoff_date=cutoff_dt,
            window_days=news_window_days,
        )

        # --- 2. Format price history ---
        ph = market.get("price_history", [])
        if ph:
            ph_lines = [f"  {p['date']}: {p['price']:.4f}" for p in ph[-15:]]  # last 15 points
            price_history_str = "\n".join(ph_lines)
        else:
            price_history_str = "  No price history available."

        price_at_cutoff = market.get("price_at_cutoff")
        price_str = f"{price_at_cutoff:.4f}" if price_at_cutoff is not None else "N/A"

        # --- 3. Build prompt ---
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question,
            answer_yes=market.get("answer_yes", "Yes"),
            answer_no=market.get("answer_no", "No"),
            cutoff_date=cutoff_date_s,
            price_at_cutoff=price_str,
            price_history=price_history_str,
            news_context=news_text,
        )

        # --- 4. Generate text answer ---
        print(f"  [Qwen] Generating answer...")
        model_text = self.generate(SYSTEM_PROMPT, user_prompt)

        # --- 5. Extract log-probs ---
        # Build the full prompt text including the "Answer:" prefix so the model
        # scores the next token (Yes/No) after that prefix
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
            # Prime with the model's own text + "Answer:" so log-probs are at decision point
            {"role": "assistant", "content": model_text.rstrip() + "\nAnswer:"},
        ]
        full_prompt_for_logprobs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        print(f"  [Qwen] Extracting log-probs...")
        lp_result = self._extract_yes_no_logprobs(full_prompt_for_logprobs)

        # --- 6. Parse text label as secondary signal ---
        text_lower = model_text.lower()
        if "answer: yes" in text_lower or text_lower.strip().endswith("yes"):
            parsed_text_label = 1
        elif "answer: no" in text_lower or text_lower.strip().endswith("no"):
            parsed_text_label = 0
        else:
            # fallback: use log-prob decision
            parsed_text_label = 1 if lp_result["yes_prob"] >= 0.5 else 0

        return {
            "market_id":          market["market_id"],
            "question":           question,
            "category":           market.get("category", "unknown"),
            "ground_truth":       market["ground_truth"],
            "predicted_prob_yes": lp_result["yes_prob"],
            "predicted_label":    1 if lp_result["yes_prob"] >= 0.5 else 0,
            "model_text_answer":  model_text,
            "parsed_text_label":  parsed_text_label,
            "yes_logprob":        lp_result["yes_logprob"],
            "no_logprob":         lp_result["no_logprob"],
            "news_context_used":  news_text[:500] + "..." if len(news_text) > 500 else news_text,
        }
