"""
Polymarket Dataset Processor
=============================
Converts polymarket_dataset.csv into the parquet format expected by
Search-R1's veRL training pipeline (same schema as nq_search.py).

Each row becomes:
  {
    "data_source": "polymarket",
    "prompt": [{"role": "user", "content": "<instruction + question>"}],
    "ability": "fact-reasoning",
    "reward_model": {"style": "rule", "ground_truth": {"target": ["Yes"]}},
    "extra_info": {"split": "train", "index": 0, "end_date": "2025-03-01", "market_id": "..."}
  }

Only binary Yes/No markets are kept. Rows with other labels (Up/Down/$X) are dropped.

Usage:
    python training/data/process_polymarket.py \
        --csv polymarket_dataset.csv \
        --out  training/data/polymarket_search \
        --val-ratio 0.1
"""

import argparse
import csv
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# -----------------------------------------------------------------------
# Search-R1 instruction prompt (base model format — same as nq_search.py)
# -----------------------------------------------------------------------
INSTRUCTION = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as you want. "
    "If you find no further external knowledge needed, you can directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. "
    "For a Yes/No prediction market question, your answer must be exactly: "
    "<answer> Yes </answer> or <answer> No </answer>."
)


def make_prompt(question: str, description: str) -> str:
    """Build the user prompt string fed to the model."""
    q = question.strip()
    if not q.endswith("?"):
        q += "?"

    if description and description.strip():
        desc = description.strip()[:600]
        return (
            f"{INSTRUCTION}\n\n"
            f"Question: {q}\n\n"
            f"Resolution criteria: {desc}"
        )
    return f"{INSTRUCTION}\n\nQuestion: {q}"


def parse_end_date(end_date_str: str) -> str:
    """Return YYYY-MM-DD string from ISO datetime."""
    try:
        dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return end_date_str[:10]


def load_and_filter(csv_path: str) -> list[dict]:
    """Load CSV, drop non-binary markets, return cleaned rows."""
    rows = []
    dropped_non_binary = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("ground_truth_label", "").strip()
            if label not in ("Yes", "No"):
                dropped_non_binary += 1
                continue
            rows.append(row)

    print(f"  Loaded {len(rows)} binary markets ({dropped_non_binary} non-binary dropped)")
    return rows


def make_record(row: dict, idx: int, split: str) -> dict:
    question    = row["question"].strip()
    description = row.get("description", "").strip()
    label       = row["ground_truth_label"].strip()   # "Yes" or "No"
    end_date    = parse_end_date(row.get("end_date", ""))
    market_id   = row.get("market_id", str(idx))

    prompt_text = make_prompt(question, description)

    return {
        "data_source": "polymarket",
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": [label]}   # ["Yes"] or ["No"]
        },
        "extra_info": {
            "split":       split,
            "index":       idx,
            "market_id":   market_id,
            "end_date": end_date,
            "question":    question,
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default="polymarket_dataset.csv")
    parser.add_argument("--out",       default="training/data/polymarket_search",
                        help="Output directory — train.parquet and test.parquet written here")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Fraction of data held out for validation")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading {args.csv}...")
    rows = load_and_filter(args.csv)

    random.shuffle(rows)
    n_val   = max(1, int(len(rows) * args.val_ratio))
    val_rows   = rows[:n_val]
    train_rows = rows[n_val:]
    print(f"  Train: {len(train_rows)}  |  Val: {len(val_rows)}")

    # Label balance check
    for name, split_rows in [("train", train_rows), ("val", val_rows)]:
        yes = sum(1 for r in split_rows if r["ground_truth_label"] == "Yes")
        print(f"  {name}: Yes={yes}  No={len(split_rows)-yes}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, split_rows in [("train", train_rows), ("test", val_rows)]:
        records = [make_record(row, idx, split) for idx, row in enumerate(split_rows)]
        df = pd.DataFrame(records)
        out_path = out_dir / f"{split}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved {len(records)} records → {out_path}")

    print("\nDone. Run training with:")
    print(f"  bash training/configs/grpo_3b.sh")


if __name__ == "__main__":
    main()
