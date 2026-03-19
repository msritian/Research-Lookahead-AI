"""
Benchmark Evaluator
====================
Loads results from both runners, computes metrics, and outputs a
comparison report as JSON + prints a human-readable table.

Metrics computed:
  - Accuracy           (predicted_label == ground_truth)
  - Brier Score        (mean squared error of predicted_prob_yes vs ground_truth)
  - Log Loss           (cross-entropy)
  - Per-category breakdown for both metrics
  - Per-market head-to-head comparison

Usage:
    python benchmark/evaluate.py \
        --lookahead benchmark/results/lookahead_results.json \
        --searchr1  benchmark/results/searchr1_results.json \
        --out       benchmark/results/comparison_report.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def brier_score(results: List[Dict]) -> float:
    """Mean squared error of predicted_prob_yes vs ground_truth."""
    return sum(
        (r["predicted_prob_yes"] - r["ground_truth"]) ** 2
        for r in results
    ) / len(results)


def log_loss(results: List[Dict], eps: float = 1e-7) -> float:
    """Binary cross-entropy."""
    total = 0.0
    for r in results:
        p   = max(eps, min(1 - eps, r["predicted_prob_yes"]))
        y   = r["ground_truth"]
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(results)


def accuracy(results: List[Dict]) -> float:
    correct = sum(1 for r in results if r["predicted_label"] == r["ground_truth"])
    return correct / len(results)


def text_accuracy(results: List[Dict]) -> float:
    """Accuracy using the text-parsed label rather than log-prob label."""
    valid = [r for r in results if r.get("parsed_text_label", -1) != -1]
    if not valid:
        return float("nan")
    correct = sum(1 for r in valid if r["parsed_text_label"] == r["ground_truth"])
    return correct / len(valid)


def per_category_metrics(results: List[Dict]) -> Dict[str, Dict]:
    cats: Dict[str, List] = {}
    for r in results:
        cat = r.get("category", "unknown")
        cats.setdefault(cat, []).append(r)

    out = {}
    for cat, recs in cats.items():
        out[cat] = {
            "n":           len(recs),
            "accuracy":    round(accuracy(recs), 4),
            "brier_score": round(brier_score(recs), 4),
            "log_loss":    round(log_loss(recs), 4),
        }
    return out


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    la_results: List[Dict],
    sr1_results: List[Dict],
) -> Dict[str, Any]:

    # Index by market_id for head-to-head
    sr1_by_id = {r["market_id"]: r for r in sr1_results}

    head_to_head = []
    for la in la_results:
        mid = la["market_id"]
        sr1 = sr1_by_id.get(mid)
        if not sr1:
            continue

        gt         = la["ground_truth"]
        la_correct = la["predicted_label"] == gt
        sr1_correct = sr1["predicted_label"] == gt

        head_to_head.append({
            "market_id":        mid,
            "question":         la["question"][:80],
            "category":         la["category"],
            "ground_truth":     "YES" if gt == 1 else "NO",
            "lookahead": {
                "prob_yes":  la["predicted_prob_yes"],
                "label":     "YES" if la["predicted_label"] == 1 else "NO",
                "correct":   la_correct,
                "brier":     round((la["predicted_prob_yes"] - gt)**2, 4),
            },
            "searchr1": {
                "prob_yes":      sr1["predicted_prob_yes"],
                "label":         "YES" if sr1["predicted_label"] == 1 else "NO",
                "correct":       sr1_correct,
                "brier":         round((sr1["predicted_prob_yes"] - gt)**2, 4),
                "search_count":  sr1.get("search_count", 0),
            },
            "winner": (
                "lookahead" if la_correct and not sr1_correct else
                "searchr1"  if sr1_correct and not la_correct else
                "tie_correct" if la_correct and sr1_correct else
                "tie_wrong"
            ),
        })

    la_metrics = {
        "accuracy":           round(accuracy(la_results), 4),
        "text_accuracy":      round(text_accuracy(la_results), 4),
        "brier_score":        round(brier_score(la_results), 4),
        "log_loss":           round(log_loss(la_results), 4),
        "per_category":       per_category_metrics(la_results),
        "avg_elapsed_sec":    round(
            sum(r.get("elapsed_sec", 0) for r in la_results) / len(la_results), 1
        ),
    }

    sr1_metrics = {
        "accuracy":           round(accuracy(sr1_results), 4),
        "text_accuracy":      round(text_accuracy(sr1_results), 4),
        "brier_score":        round(brier_score(sr1_results), 4),
        "log_loss":           round(log_loss(sr1_results), 4),
        "per_category":       per_category_metrics(sr1_results),
        "avg_searches_used":  round(
            sum(r.get("search_count", 0) for r in sr1_results) / len(sr1_results), 2
        ),
        "avg_elapsed_sec":    round(
            sum(r.get("elapsed_sec", 0) for r in sr1_results) / len(sr1_results), 1
        ),
    }

    wins = {"lookahead": 0, "searchr1": 0, "tie_correct": 0, "tie_wrong": 0}
    for h in head_to_head:
        wins[h["winner"]] += 1

    return {
        "summary": {
            "n_markets":       len(head_to_head),
            "lookahead_wins":  wins["lookahead"],
            "searchr1_wins":   wins["searchr1"],
            "both_correct":    wins["tie_correct"],
            "both_wrong":      wins["tie_wrong"],
            "overall_winner":  (
                "lookahead" if wins["lookahead"] > wins["searchr1"] else
                "searchr1"  if wins["searchr1"] > wins["lookahead"] else
                "tie"
            ),
        },
        "lookahead_metrics":  la_metrics,
        "searchr1_metrics":   sr1_metrics,
        "head_to_head":       head_to_head,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(report: Dict):
    s   = report["summary"]
    la  = report["lookahead_metrics"]
    sr1 = report["searchr1_metrics"]

    W = 62
    print("\n" + "=" * W)
    print("  BENCHMARK COMPARISON REPORT".center(W))
    print("  LookaheadAI (Qwen2.5-3B-Instruct + Exa)".center(W))
    print("  vs Search-R1 (Qwen2.5-3B GRPO + Exa)".center(W))
    print("=" * W)

    print(f"\n{'Metric':<30} {'LookaheadAI':>14} {'Search-R1':>14}")
    print("-" * W)

    def row(label, la_val, sr1_val, lower_is_better=False, fmt=".4f"):
        la_s  = f"{la_val:{fmt}}"
        sr1_s = f"{sr1_val:{fmt}}"
        # Bold winner marker
        if lower_is_better:
            la_mark  = " ✓" if la_val  < sr1_val  else ""
            sr1_mark = " ✓" if sr1_val < la_val   else ""
        else:
            la_mark  = " ✓" if la_val  > sr1_val  else ""
            sr1_mark = " ✓" if sr1_val > la_val   else ""
        print(f"{label:<30} {la_s+la_mark:>14} {sr1_s+sr1_mark:>14}")

    row("Accuracy",           la["accuracy"],    sr1["accuracy"],    lower_is_better=False)
    row("Text Accuracy",      la["text_accuracy"], sr1["text_accuracy"], lower_is_better=False)
    row("Brier Score ↓",      la["brier_score"], sr1["brier_score"],  lower_is_better=True)
    row("Log Loss ↓",         la["log_loss"],    sr1["log_loss"],     lower_is_better=True)
    row("Avg Time (s) ↓",     la["avg_elapsed_sec"], sr1["avg_elapsed_sec"],
        lower_is_better=True, fmt=".1f")

    print(f"\n{'Per-Category Accuracy':}")
    print("-" * W)
    all_cats = sorted(set(
        list(la["per_category"].keys()) + list(sr1["per_category"].keys())
    ))
    for cat in all_cats:
        la_c  = la["per_category"].get(cat,  {}).get("accuracy", float("nan"))
        sr1_c = sr1["per_category"].get(cat, {}).get("accuracy", float("nan"))
        n     = la["per_category"].get(cat, {}).get("n", sr1["per_category"].get(cat, {}).get("n", 0))
        print(f"  {cat:<20} (n={n})  LA={la_c:.4f}  SR1={sr1_c:.4f}")

    print(f"\n{'Head-to-Head per Market':}")
    print("-" * W)
    for h in report["head_to_head"]:
        la_c   = "✓" if h["lookahead"]["correct"] else "✗"
        sr1_c  = "✓" if h["searchr1"]["correct"]  else "✗"
        print(f"  [{h['category'][:8]:8}] {h['question'][:42]}")
        print(f"           GT={h['ground_truth']:3}  "
              f"LA={h['lookahead']['label']:3}({h['lookahead']['prob_yes']:.2f}){la_c}  "
              f"SR1={h['searchr1']['label']:3}({h['searchr1']['prob_yes']:.2f}){sr1_c}  "
              f"searches={h['searchr1']['search_count']}")

    print(f"\n{'Overall':}")
    print("-" * W)
    print(f"  LookaheadAI wins:  {s['lookahead_wins']}")
    print(f"  Search-R1 wins:    {s['searchr1_wins']}")
    print(f"  Both correct:      {s['both_correct']}")
    print(f"  Both wrong:        {s['both_wrong']}")
    print(f"\n  WINNER: {s['overall_winner'].upper()}")
    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lookahead", default="benchmark/results/lookahead_results.json")
    p.add_argument("--searchr1",  default="benchmark/results/searchr1_results.json")
    p.add_argument("--out",       default="benchmark/results/comparison_report.json")
    return p.parse_args()


def main():
    args = parse_args()

    la_path  = Path(args.lookahead)
    sr1_path = Path(args.searchr1)

    if not la_path.exists():
        print(f"ERROR: {la_path} not found. Run run_lookahead.py first.")
        sys.exit(1)
    if not sr1_path.exists():
        print(f"ERROR: {sr1_path} not found. Run run_searchr1.py first.")
        sys.exit(1)

    la_results  = json.loads(la_path.read_text())
    sr1_results = json.loads(sr1_path.read_text())

    print(f"Loaded {len(la_results)} LookaheadAI results")
    print(f"Loaded {len(sr1_results)} Search-R1 results")

    report = build_report(la_results, sr1_results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved report → {out_path}")

    print_report(report)


if __name__ == "__main__":
    main()
