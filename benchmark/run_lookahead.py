"""
LookaheadAI Benchmark Runner
=============================
Runs the LookaheadAI method on all markets in benchmark_markets.json:
  - Uses QwenProvider (Qwen2.5-3B-Instruct, local)
  - Fetches Exa news up to cutoff_date (7 days before resolution)
  - Extracts log-prob calibrated Yes/No probability
  - Saves results to benchmark/results/lookahead_results.json

Usage:
    python benchmark/run_lookahead.py [--markets PATH] [--out PATH]
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone

# Ensure repo root is on path when run from Colab
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agents.qwen_provider import QwenProvider


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--markets", default=str(REPO_ROOT / "benchmark/data/benchmark_markets.json"))
    p.add_argument("--out",     default=str(REPO_ROOT / "benchmark/results/lookahead_results.json"))
    p.add_argument("--exa-key", default=None, help="Exa API key (or set EXA_API_KEY env var)")
    p.add_argument("--news-window-days", type=int, default=30)
    p.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit (saves VRAM)")
    return p.parse_args()


def main():
    args = parse_args()

    exa_key = args.exa_key or os.environ.get("EXA_API_KEY", "")
    if not exa_key:
        print("WARNING: EXA_API_KEY not set — news context will be empty.")

    markets_path = Path(args.markets)
    if not markets_path.exists():
        print(f"ERROR: {markets_path} not found. Run prepare_dataset.py first.")
        sys.exit(1)

    markets = json.loads(markets_path.read_text())
    print(f"Loaded {len(markets)} markets from {markets_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    provider = QwenProvider(
        exa_api_key=exa_key,
        load_in_4bit=args.load_4bit,
    )

    results = []
    for i, market in enumerate(markets):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(markets)}] {market['question'][:70]}")
        print(f"  Category: {market['category']} | Ground truth: {'YES' if market['ground_truth']==1 else 'NO'}")
        t0 = time.time()

        try:
            result = provider.predict_market(
                market,
                news_window_days=args.news_window_days,
            )
            result["elapsed_sec"] = round(time.time() - t0, 1)
            result["method"] = "lookahead_qwen"
            print(f"  → prob_yes={result['predicted_prob_yes']:.3f}  "
                  f"label={'YES' if result['predicted_label']==1 else 'NO'}  "
                  f"gt={'YES' if market['ground_truth']==1 else 'NO'}  "
                  f"({result['elapsed_sec']}s)")
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
                "parsed_text_label":  0,
                "yes_logprob":        None,
                "no_logprob":         None,
                "news_context_used":  "",
                "method":             "lookahead_qwen",
                "elapsed_sec":        round(time.time() - t0, 1),
                "error":              str(e),
            }

        results.append(result)

        # Save incrementally after each market
        out_path.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}")
    print(f"Done. Saved {len(results)} results → {out_path}")

    # Quick summary
    correct = sum(1 for r in results if r["predicted_label"] == r["ground_truth"])
    print(f"Accuracy: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")

    brier = sum((r["predicted_prob_yes"] - r["ground_truth"])**2 for r in results) / len(results)
    print(f"Brier Score: {brier:.4f}  (lower is better, 0=perfect)")


if __name__ == "__main__":
    main()
