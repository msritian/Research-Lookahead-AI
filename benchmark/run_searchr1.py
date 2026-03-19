"""
Search-R1 Benchmark Runner
===========================
Runs the Search-R1 method on all markets in benchmark_markets.json:
  - Loads the pretrained checkpoint: PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo
  - Uses the exact Search-R1 multi-turn inference loop (think → search → information → answer)
  - Retrieval is backed by the Exa retriever server (benchmark/exa_retriever_server.py)
  - Extracts log-probs over "Yes"/"No" after the <answer> tag for calibration
  - Saves results to benchmark/results/searchr1_results.json

Prerequisites:
  - Exa retriever server running: python benchmark/exa_retriever_server.py
  - Or set RETRIEVER_URL env var to override (default: http://127.0.0.1:8000/retrieve)

Usage:
    python benchmark/run_searchr1.py [--markets PATH] [--out PATH]
"""

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEARCHR1_MODEL  = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo"
DEFAULT_RETRIEVER_URL = "http://127.0.0.1:8000/retrieve"
MAX_TURNS       = 5      # maximum search rounds per question
MAX_NEW_TOKENS  = 1024   # tokens per generation step — matches infer.py
TOPK            = 3      # documents to retrieve per search

# Search-R1 XML tag protocol
SEARCH_START  = "<search>"
SEARCH_END    = "</search>"
ANSWER_START  = "<answer>"
ANSWER_END    = "</answer>"
INFO_START    = "<information>"
INFO_END      = "</information>"

# Instruction prompt matching Search-R1 training data format
INSTRUCTION = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "For a Yes/No prediction market question, your answer must be exactly: "
    "<answer> Yes </answer> or <answer> No </answer>."
)

YES_TOKENS = ["Yes", "YES", "yes"]
NO_TOKENS  = ["No",  "NO",  "no"]


# ---------------------------------------------------------------------------
# Search-R1 inference helpers
# ---------------------------------------------------------------------------

def call_retriever(query: str, retriever_url: str, topk: int = TOPK) -> str:
    """
    Calls the /retrieve endpoint and formats results exactly as Search-R1's
    _passages2string() does in infer.py:
        Doc 1(Title: "Title") text
    """
    try:
        payload  = {"queries": [query], "topk": topk, "return_scores": True}
        response = requests.post(retriever_url, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        docs = data.get("result", [[]])[0]

        if not docs:
            return "No results found."

        format_reference = ""
        for idx, doc_item in enumerate(docs):
            content = doc_item.get("document", {}).get("contents", "")
            # Exa server stores as: '"Title"\ntext' — matches Search-R1 corpus format
            lines = content.split("\n")
            title = lines[0]                        # e.g. '"Some Title"'
            text  = "\n".join(lines[1:])[:800]      # body, capped
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference.strip()

    except Exception as e:
        return f"Search error: {e}"


def parse_search_query(text: str) -> Optional[str]:
    """Extracts the query from <search>query</search>."""
    m = re.search(r"<search>(.*?)</search>", text, re.S)
    return m.group(1).strip() if m else None


def parse_answer(text: str) -> Optional[str]:
    """Extracts the answer from <answer>answer</answer>."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    return m.group(1).strip() if m else None


class SearchR1Runner:
    def __init__(
        self,
        model_name: str = SEARCHR1_MODEL,
        retriever_url: str = DEFAULT_RETRIEVER_URL,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
    ):
        self.model_name    = model_name
        self.retriever_url = retriever_url
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_4bit  = load_in_4bit
        self._model        = None
        self._tokenizer    = None

    def _load_model(self):
        if self._model is not None:
            return

        print(f"[SearchR1] Loading {self.model_name} on {self.device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
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

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self._model.eval()
        print(f"[SearchR1] Model loaded.")

    def _generate_step(self, prompt_text: str, stop_strings: List[str]) -> Tuple[str, bool]:
        """
        Generates tokens until a stop string is found or max_new_tokens.
        Returns (generated_text, hit_stop).
        """
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        # Use StoppingCriteria for stop strings
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnString(StoppingCriteria):
            def __init__(self, stop_seqs: List[str], tokenizer):
                self.stop_ids = [
                    tokenizer.encode(s, add_special_tokens=False)
                    for s in stop_seqs
                ]

            def __call__(self, input_ids, scores, **kwargs):
                last_tokens = input_ids[0, -10:].tolist()
                seq = input_ids[0].tolist()
                for stop in self.stop_ids:
                    n = len(stop)
                    if seq[-n:] == stop:
                        return True
                return False

        stopper = StopOnString(stop_strings, self._tokenizer)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,      # matches infer.py: stochastic rollout as trained
                temperature=0.7,     # matches infer.py exactly
                pad_token_id=self._tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([stopper]),
            )

        new_ids = output_ids[0][input_len:]
        generated = self._tokenizer.decode(new_ids, skip_special_tokens=False)

        # Check if we stopped at one of the stop strings
        hit_stop = any(s in generated for s in stop_strings)
        return generated, hit_stop

    def _get_logprobs_yes_no(self, prompt_text: str) -> Dict[str, float]:
        """
        Score log-probs of Yes/No tokens as the next token after prompt_text.
        """
        inputs   = self._tokenizer(prompt_text, return_tensors="pt").to(self.device)

        def score_token(token_str: str) -> float:
            tids = self._tokenizer.encode(token_str, add_special_tokens=False)
            if not tids:
                return float("-inf")
            lp_sum = 0.0
            cur = inputs["input_ids"].clone()
            with torch.no_grad():
                for tid in tids:
                    out = self._model(input_ids=cur)
                    lps = torch.nn.functional.log_softmax(out.logits[0, -1, :], dim=-1)
                    lp_sum += lps[tid].item()
                    cur = torch.cat([cur, torch.tensor([[tid]], device=self.device)], dim=1)
            return lp_sum / len(tids)

        yes_lp = max(score_token(t) for t in YES_TOKENS)
        no_lp  = max(score_token(t) for t in NO_TOKENS)

        yes_p = math.exp(yes_lp)
        no_p  = math.exp(no_lp)
        total = yes_p + no_p + 1e-12

        return {
            "yes_logprob": round(yes_lp, 6),
            "no_logprob":  round(no_lp, 6),
            "yes_prob":    round(yes_p / total, 6),
        }

    def run_inference(self, question: str) -> Dict[str, Any]:
        """
        Full Search-R1 multi-turn inference loop for one question.
        Mirrors infer.py exactly:
          - apply_chat_template on the initial prompt (Qwen2.5 requires this)
          - stop at </search> or </answer> each turn
          - inject search results as: \n\n{output_text}<information>{results}</information>\n\n
          - loop until </answer> hit or EOS
        """
        self._load_model()

        # Build initial prompt string — matches infer.py exactly
        raw_prompt = (
            f"{INSTRUCTION}\n"
            f"Question: {question}\n"
        )

        # Apply chat template if the tokenizer has one (Qwen2.5 does) — matches infer.py
        if self._tokenizer.chat_template:
            full_prompt = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            full_prompt = raw_prompt

        trajectory        = []
        final_answer_text = None
        search_count      = 0

        # EOS token ids for Qwen2.5 (matches infer.py: curr_eos = [151645, 151643])
        eos_ids = set(self._tokenizer.convert_tokens_to_ids(
            [self._tokenizer.eos_token, "<|im_end|>"]
        ) + [151645, 151643])

        while True:
            # Generate until </search>, </answer>, or EOS — matches infer.py while True loop
            generated, hit_stop = self._generate_step(
                full_prompt,
                stop_strings=[SEARCH_END, ANSWER_END]
            )

            # Check for EOS (model naturally stopped without structured tag)
            output_ids = self._tokenizer.encode(generated, add_special_tokens=False)
            hit_eos = bool(output_ids and output_ids[-1] in eos_ids)

            if hit_eos or (not hit_stop and not generated.strip()):
                # Model ended naturally — extract answer if present
                final_answer_text = parse_answer(full_prompt + generated) or generated.strip()
                trajectory.append({"turn": search_count, "type": "eos", "text": generated})
                full_prompt += generated
                break

            full_prompt += generated

            # --- Model gave an answer ---
            if ANSWER_END in generated:
                final_answer_text = parse_answer(generated)
                trajectory.append({"turn": search_count, "type": "answer", "text": generated})
                break

            # --- Model issued a search ---
            query = parse_search_query(generated)
            if query:
                search_count += 1
                print(f"    [SearchR1] Search #{search_count}: {query[:60]}")
                search_results = call_retriever(query, self.retriever_url)

                # Inject exactly as infer.py: '\n\n{output_text}<information>{results}</information>\n\n'
                # output_text is already in full_prompt; we append the information block
                info_block  = f"\n\n<information>{search_results}</information>\n\n"
                full_prompt += info_block

                trajectory.append({
                    "turn":    search_count,
                    "type":    "search",
                    "query":   query,
                    "results": search_results[:300],
                })

                # Safety cap — avoid infinite loop
                if search_count >= MAX_TURNS:
                    break
                continue

            # Invalid output — inject error message matching infer.py behaviour
            error_msg = (
                "\nMy previous action is invalid. "
                "If I want to search, I should put the query between <search> and </search>. "
                "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
                "Let me try again.\n"
            )
            full_prompt += error_msg
            trajectory.append({"turn": search_count, "type": "invalid", "text": generated})

        # --- Extract log-probs at the answer decision point ---
        # Build context up to and including <answer> so the model scores Yes/No next
        if ANSWER_START in full_prompt:
            idx          = full_prompt.rfind(ANSWER_START)
            answer_prime = full_prompt[:idx + len(ANSWER_START)] + " "
        else:
            answer_prime = full_prompt.rstrip() + f"\n{ANSWER_START} "

        lp = self._get_logprobs_yes_no(answer_prime)

        return {
            "final_answer_text": final_answer_text or "",
            "search_count":      search_count,
            "trajectory":        trajectory,
            "yes_logprob":       lp["yes_logprob"],
            "no_logprob":        lp["no_logprob"],
            "yes_prob":          lp["yes_prob"],
        }


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--markets",       default=str(REPO_ROOT / "benchmark/data/benchmark_markets.json"))
    p.add_argument("--out",           default=str(REPO_ROOT / "benchmark/results/searchr1_results.json"))
    p.add_argument("--retriever-url", default=os.environ.get("RETRIEVER_URL", DEFAULT_RETRIEVER_URL))
    p.add_argument("--load-4bit",     action="store_true")
    return p.parse_args()


def label_from_text(text: str) -> int:
    """Converts a free-form text answer to 1 (yes) or 0 (no)."""
    t = text.lower().strip()
    if t.startswith("yes") or t == "yes":
        return 1
    if t.startswith("no") or t == "no":
        return 0
    # Fallback: presence check
    if "yes" in t and "no" not in t:
        return 1
    if "no" in t and "yes" not in t:
        return 0
    return -1  # ambiguous


def main():
    args = parse_args()

    markets_path = Path(args.markets)
    if not markets_path.exists():
        print(f"ERROR: {markets_path} not found. Run prepare_dataset.py first.")
        sys.exit(1)

    markets = json.loads(markets_path.read_text())
    print(f"Loaded {len(markets)} markets from {markets_path}")

    # Check retriever health
    try:
        r = requests.get(args.retriever_url.replace("/retrieve", "/health"), timeout=5)
        print(f"Retriever health: {r.json()}")
    except Exception as e:
        print(f"WARNING: Retriever not reachable at {args.retriever_url}: {e}")
        print("  → Start it with: python benchmark/exa_retriever_server.py")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runner = SearchR1Runner(
        retriever_url=args.retriever_url,
        load_in_4bit=args.load_4bit,
    )

    results = []
    for i, market in enumerate(markets):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(markets)}] {market['question'][:70]}")
        print(f"  Category: {market['category']} | Ground truth: {'YES' if market['ground_truth']==1 else 'NO'}")
        t0 = time.time()

        try:
            inference = runner.run_inference(market["question"])
            elapsed   = round(time.time() - t0, 1)

            text_label = label_from_text(inference["final_answer_text"])
            # If log-prob and text disagree, log-prob takes precedence
            logprob_label = 1 if inference["yes_prob"] >= 0.5 else 0

            print(f"  → prob_yes={inference['yes_prob']:.3f}  "
                  f"text='{inference['final_answer_text'][:20]}'  "
                  f"searches={inference['search_count']}  "
                  f"gt={'YES' if market['ground_truth']==1 else 'NO'}  "
                  f"({elapsed}s)")

            result = {
                "market_id":          market["market_id"],
                "question":           market["question"],
                "category":           market.get("category"),
                "ground_truth":       market["ground_truth"],
                "predicted_prob_yes": inference["yes_prob"],
                "predicted_label":    logprob_label,
                "model_text_answer":  inference["final_answer_text"],
                "parsed_text_label":  text_label,
                "yes_logprob":        inference["yes_logprob"],
                "no_logprob":         inference["no_logprob"],
                "search_count":       inference["search_count"],
                "trajectory":         inference["trajectory"],
                "method":             "searchr1",
                "elapsed_sec":        elapsed,
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            result = {
                "market_id":          market["market_id"],
                "question":           market["question"],
                "category":           market.get("category"),
                "ground_truth":       market["ground_truth"],
                "predicted_prob_yes": 0.5,
                "predicted_label":    0,
                "model_text_answer":  f"ERROR: {e}",
                "parsed_text_label":  -1,
                "yes_logprob":        None,
                "no_logprob":         None,
                "search_count":       0,
                "trajectory":         [],
                "method":             "searchr1",
                "elapsed_sec":        round(time.time() - t0, 1),
                "error":              str(e),
            }

        results.append(result)
        out_path.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}")
    print(f"Done. Saved {len(results)} results → {out_path}")

    correct = sum(1 for r in results if r["predicted_label"] == r["ground_truth"])
    brier   = sum((r["predicted_prob_yes"] - r["ground_truth"])**2 for r in results) / len(results)
    print(f"Accuracy:    {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
    print(f"Brier Score: {brier:.4f}  (lower is better)")


if __name__ == "__main__":
    main()
