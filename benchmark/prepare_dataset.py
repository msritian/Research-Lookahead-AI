"""
Benchmark Dataset Sampler
=========================
Downloads a slice of SII-WANGZJ/Polymarket_data from HuggingFace,
filters for resolved markets with a clear winner, picks 10 balanced rows
across categories, and saves benchmark_markets.json.

Run once on Colab before running benchmark.py:
    python benchmark/prepare_dataset.py
"""

import json
import re
import random
import time
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Category classifier — maps market question keywords to broad buckets
# ---------------------------------------------------------------------------
CATEGORY_RULES = [
    ("crypto",   re.compile(r"\b(bitcoin|btc|ethereum|eth|solana|sol|xrp|crypto|coin|blockchain)\b", re.I)),
    ("politics", re.compile(r"\b(president|election|congress|senate|vote|party|democrat|republican|candidate|governor|mayor|ballot|white house|parliament|prime minister)\b", re.I)),
    ("sports",   re.compile(r"\b(nba|nfl|nhl|mlb|soccer|football|basketball|baseball|hockey|tennis|championship|playoffs|world cup|super bowl|fifa|uefa|nba|espn|spread|over/under)\b", re.I)),
    ("finance",  re.compile(r"\b(stock|nasdaq|s&p|dow|fed|federal reserve|interest rate|inflation|gdp|market cap|ipo|earnings|bond|yield|recession)\b", re.I)),
    ("geopolitics", re.compile(r"\b(war|ceasefire|nato|russia|ukraine|china|taiwan|israel|gaza|iran|nuclear|sanctions|treaty|military)\b", re.I)),
    ("science",  re.compile(r"\b(fda|vaccine|drug|clinical|trial|nasa|space|ai|artificial intelligence|model|gpt|openai|approval|patent)\b", re.I)),
]

TARGET_PER_CATEGORY = 2   # aim for 2 per bucket (6 buckets * 2 = 12, we take top 10)
TOTAL_TARGET        = 10
RANDOM_SEED         = 42
MIN_VOLUME_USD      = 5_000   # raised from 100 — filters out micro/noise markets

# Patterns that indicate noise markets not useful for forecasting benchmarks:
# sports spreads/totals, intraday crypto up/down, minute-level events
NOISE_PATTERNS = re.compile(
    r"\b(spread|over/under|o/u|\+[0-9]|\-[0-9]|up or down|total kills|total points|"
    r"run line|puck line|moneyline|\bml\b|1st half|2nd half|"
    r"[0-9]+am et|[0-9]+pm et|[0-9]+am pst|[0-9]+pm pst)\b",
    re.I,
)

# Qwen2.5 was released September 2024 with a training cutoff of ~early 2024.
# To guarantee zero pretraining leakage of outcomes, we only include markets
# that resolved AFTER this date — the model cannot have seen their results.
LEAKAGE_SAFE_CUTOFF = "2025-01-01"  # markets ending on or after this date are safe


GAMMA_API = "https://gamma-api.polymarket.com/markets"


def fetch_gamma_market(slug: str, retries: int = 3) -> dict:
    """
    Fetches the full Gamma API market record for a slug.
    Returns a dict with at minimum 'description' and 'outcomePrices' keys (empty if fetch fails).
    """
    url = f"{GAMMA_API}?slug={urllib.parse.quote(slug)}&limit=1"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "benchmark-fetcher/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                if data and isinstance(data, list):
                    return data[0]
                return {}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                print(f"    [warn] Gamma API fetch failed for '{slug}': {e}")
                return {}
    return {}


def fetch_resolution_criteria(slug: str, retries: int = 3) -> str:
    """Returns the resolution criteria description for a slug, or empty string."""
    gamma = fetch_gamma_market(slug, retries=retries)
    return gamma.get("description", "").strip()


def classify(question: str) -> str:
    for cat, pattern in CATEGORY_RULES:
        if pattern.search(question):
            return cat
    return "other"


def parse_outcome(outcome_prices_str: str) -> Optional[int]:
    """
    Returns 1 if token1 (YES) won, 0 if token2 (NO) won, None if ambiguous.
    outcome_prices is stored as a JSON-like string e.g. "['0.99', '0.01']"
    """
    try:
        cleaned = outcome_prices_str.replace("'", '"')
        prices = json.loads(cleaned)
        p1, p2 = float(prices[0]), float(prices[1])
        if p1 >= 0.95:
            return 1
        if p2 >= 0.95:
            return 0
    except Exception:
        pass
    return None



def main():
    import pandas as pd
    from datasets import load_dataset

    output_path = Path(__file__).parent / "data" / "benchmark_markets.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)

    print("Loading markets.parquet from HuggingFace (this may take a minute)...")
    markets_ds = load_dataset(
        "SII-WANGZJ/Polymarket_data",
        data_files="markets.parquet",
        split="train",
        streaming=False,
    )
    markets_df = markets_ds.to_pandas()
    print(f"  Loaded {len(markets_df):,} market rows")

    # --- Filter: resolved + clear winner ---
    resolved = markets_df[markets_df["closed"] == 1].copy()
    print(f"  Closed markets: {len(resolved):,}")

    resolved["ground_truth"] = resolved["outcome_prices"].apply(
        lambda x: parse_outcome(str(x)) if x is not None else None
    )
    resolved = resolved[resolved["ground_truth"].notna()].copy()
    print(f"  Markets with clear winner: {len(resolved):,}")

    # Exclude low volume — filters noise/micro markets
    resolved = resolved[resolved["volume"] >= MIN_VOLUME_USD].copy()
    print(f"  After volume filter (>=${MIN_VOLUME_USD:,} USD): {len(resolved):,}")

    # Exclude noise markets: sports spreads/totals, intraday crypto, etc.
    resolved = resolved[~resolved["question"].apply(lambda q: bool(NOISE_PATTERNS.search(str(q))))].copy()
    print(f"  After noise filter: {len(resolved):,}")

    # --- Leakage guard: only markets that resolved AFTER model training cutoff ---
    # Qwen2.5 training data cutoff is ~early 2024 (released Sep 2024).
    # Markets resolving after Jan 2025 cannot have their outcomes in pretraining.
    import pandas as pd
    cutoff_dt = pd.Timestamp(LEAKAGE_SAFE_CUTOFF, tz="UTC")
    if resolved["end_date"].dt.tz is None:
        resolved["end_date"] = resolved["end_date"].dt.tz_localize("UTC")
    resolved = resolved[resolved["end_date"] >= cutoff_dt].copy()
    print(f"  After leakage-safe cutoff (end_date >= {LEAKAGE_SAFE_CUTOFF}): {len(resolved):,}")

    # --- Classify ---
    resolved["category"] = resolved["question"].apply(classify)

    # --- Balanced sampling ---
    selected = []
    counts = {}
    shuffled = resolved.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    for _, row in shuffled.iterrows():
        cat = row["category"]
        if counts.get(cat, 0) < TARGET_PER_CATEGORY:
            selected.append(row)
            counts[cat] = counts.get(cat, 0) + 1
        if len(selected) >= TOTAL_TARGET:
            break

    # If under target, fill with "other"
    if len(selected) < TOTAL_TARGET:
        remaining = shuffled[~shuffled.index.isin([r.name for r in selected])]
        for _, row in remaining.iterrows():
            selected.append(row)
            if len(selected) >= TOTAL_TARGET:
                break

    print(f"\nSelected {len(selected)} markets:")
    for row in selected:
        print(f"  [{row['category']:12s}] {row['question'][:80]}")

    # --- Build final benchmark records ---
    # trades.parquet is 32GB and sorted by time — post-2025 markets are at the very end.
    # Skip it entirely. price_at_cutoff comes from Gamma API lastTradePrice instead,
    # which is the final settlement price and is always present for resolved markets.
    print("\nFetching Gamma API data (resolution criteria + crowd price) for each market...")
    records = []
    for i, row in enumerate(selected):
        market_id   = str(row["id"])
        slug        = str(row.get("slug", ""))
        end_date_ts = int(row["end_date"].timestamp()) if hasattr(row["end_date"], "timestamp") else 0
        cutoff_ts   = end_date_ts - (7 * 86400)  # 7 days before resolution

        print(f"  [{i+1}/{len(selected)}] {str(row['question'])[:65]}...")
        gamma_data = fetch_gamma_market(slug) if slug else {}

        resolution_criteria = gamma_data.get("description", "").strip()
        print(f"    resolution criteria: {len(resolution_criteria)} chars" if resolution_criteria else "    – no resolution criteria")

        # lastTradePrice is NOT used — for resolved markets it equals the settlement
        # price (~1.0 or ~0.0) which directly leaks the ground truth to the model.
        # We have no reliable way to get the price exactly at cutoff (7 days before
        # resolution) without the full trades history, so we omit it entirely.
        price_at_cutoff = None

        record = {
            "market_id":           market_id,
            "question":            str(row["question"]),
            "slug":                slug,
            "category":            str(row["category"]),
            "answer_yes":          str(row["answer1"]),
            "answer_no":           str(row["answer2"]),
            "ground_truth":        int(row["ground_truth"]),
            "volume_usd":          float(row["volume"]) if row["volume"] is not None else 0.0,
            "end_date":            row["end_date"].strftime("%Y-%m-%d") if hasattr(row["end_date"], "strftime") else str(row["end_date"]),
            "cutoff_date":         datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).strftime("%Y-%m-%d") if cutoff_ts > 0 else None,
            "price_at_cutoff":     price_at_cutoff,
            "price_history":       [],   # trades.parquet skipped — too large, wrong sort order
            "event_title":         str(row.get("event_title", "")),
            "resolution_criteria": resolution_criteria,
        }
        records.append(record)

    output_path.write_text(json.dumps(records, indent=2))
    print(f"\nSaved {len(records)} benchmark markets → {output_path}")

    # Print category summary
    from collections import Counter
    cats = Counter(r["category"] for r in records)
    print("\nCategory breakdown:")
    for cat, n in cats.most_common():
        print(f"  {cat:15s}: {n}")

    print("\nGround truth breakdown:")
    yes_count = sum(1 for r in records if r["ground_truth"] == 1)
    print(f"  YES won: {yes_count}/10  |  NO won: {10 - yes_count}/10")


if __name__ == "__main__":
    main()
