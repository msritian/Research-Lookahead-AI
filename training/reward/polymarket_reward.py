"""
Polymarket Reward Function
===========================
Drop-in replacement for Search-R1's verl/utils/reward_score/qa_em.py,
adapted for binary Yes/No prediction market questions.

Reward components:
  r_accuracy  (weight 0.6): 1.0 if correct Yes/No, 0.05 if structured but wrong, 0 if no tag
  r_temporal  (weight 0.2): fraction of retrieved articles published BEFORE end_date
  r_coverage  (weight 0.2): 1.0 if #sources used is within [min_sources, max_sources],
                            linearly penalised outside that range
                            default range: 5–10 sources (configurable via extra_info or env)

Place at: verl/utils/reward_score/polymarket_em.py
"""

import re
import random
import string
import os
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------
W_ACCURACY = 0.6
W_TEMPORAL = 0.2
W_COVERAGE = 0.2

# Default source count range (can be overridden via extra_info or env vars)
DEFAULT_MIN_SOURCES  = int(os.environ.get("REWARD_MIN_SOURCES",   5))
DEFAULT_MAX_SOURCES  = int(os.environ.get("REWARD_MAX_SOURCES",  10))
# Max age of a retrieved source relative to end_date (in days)
DEFAULT_MAX_PAST_DAYS = int(os.environ.get("REWARD_MAX_PAST_DAYS", 365))


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_solution(solution_str: str) -> Optional[str]:
    """Extract content of the last <answer>...</answer> block."""
    matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].strip()


def parse_yes_no(raw: Optional[str]) -> Optional[str]:
    """Return canonical 'Yes' or 'No', or None if ambiguous."""
    if raw is None:
        return None
    norm = normalize_answer(raw)
    if norm in ("yes",) or norm.startswith("yes"):
        return "Yes"
    if norm in ("no",) or norm.startswith("no"):
        return "No"
    return None


# ---------------------------------------------------------------------------
# Temporal reward
# ---------------------------------------------------------------------------

def compute_temporal_reward(extra_info: dict) -> float:
    """
    Returns a score in [0, 1] based on the publication dates of retrieved articles.

    A source is considered VALID if:
      - published BEFORE end_date  (no future leakage)
      - published AFTER  end_date - max_past_days  (not too stale)

    Both bounds are checked in the reward. Only the future bound is enforced
    at retrieval time (via Exa end_date). Stale sources are penalised here only.

    extra_info keys:
      - end_date:       str  YYYY-MM-DD  (market resolution date)
      - retrieved_dates: List[str | None]  (pub dates of all retrieved docs)
      - max_past_days:  int  (default 365) — sources older than this many days
                        before end_date are considered too stale

    Returns 1.0 if end_date or retrieved_dates not available (no penalty).
    """
    end_date_str    = extra_info.get("end_date")
    retrieved_dates = extra_info.get("retrieved_dates", [])
    max_past_days   = int(extra_info.get("max_past_days", DEFAULT_MAX_PAST_DAYS))

    if not end_date_str or not retrieved_dates:
        return 1.0

    try:
        target_dt = datetime.strptime(end_date_str[:10], "%Y-%m-%d")
    except Exception:
        return 1.0

    from datetime import timedelta
    earliest_allowed = target_dt - timedelta(days=max_past_days)

    valid = 0
    total = 0
    for d in retrieved_dates:
        if d is None:
            continue
        total += 1
        try:
            pub_dt = datetime.strptime(str(d)[:10], "%Y-%m-%d")
            if earliest_allowed <= pub_dt <= target_dt:
                valid += 1
            # else: either too old (< earliest_allowed) or future (> target_dt)
        except Exception:
            pass

    if total == 0:
        return 1.0

    return valid / total


# ---------------------------------------------------------------------------
# Coverage reward — number of sources used
# ---------------------------------------------------------------------------

def compute_coverage_reward(extra_info: dict) -> float:
    """
    Rewards the model for using an appropriate number of web sources.

    Score = 1.0 if n_sources in [min_sources, max_sources]
    Score linearly decays to 0 outside that range:
      - Too few: score = n_sources / min_sources
      - Too many: score = max_sources / n_sources

    Args:
        extra_info: dict with optional keys:
          - retrieved_dates: List — proxy for number of sources used
            (each retrieved doc = one source, accumulated across all turns)
          - min_sources: int (default: DEFAULT_MIN_SOURCES)
          - max_sources: int (default: DEFAULT_MAX_SOURCES)

    Returns 1.0 if retrieved_dates not available (no penalty).
    """
    retrieved_dates = extra_info.get("retrieved_dates", [])
    if not retrieved_dates:
        return 1.0   # no data — no penalty

    n_sources = len(retrieved_dates)
    min_s = int(extra_info.get("min_sources", DEFAULT_MIN_SOURCES))
    max_s = int(extra_info.get("max_sources", DEFAULT_MAX_SOURCES))

    if min_s <= n_sources <= max_s:
        return 1.0
    elif n_sources < min_s:
        # Too few sources — linear decay from 0 to 1 as n approaches min_s
        return n_sources / min_s
    else:
        # Too many sources — linear decay from 1 toward 0 as n grows past max_s
        return max_s / n_sources


# ---------------------------------------------------------------------------
# Accuracy reward
# ---------------------------------------------------------------------------

def compute_accuracy_reward(
    solution_str: str,
    ground_truth: dict,
    format_score: float = 0.05,
    score: float = 1.0,
) -> float:
    targets = ground_truth.get("target", [])
    if isinstance(targets, str):
        targets = [targets]
    canonical_gt = parse_yes_no(targets[0]) if targets else None

    raw_answer    = extract_solution(solution_str)
    parsed_answer = parse_yes_no(raw_answer)

    if raw_answer is None:
        return 0.0
    if parsed_answer is None:
        return format_score
    if parsed_answer == canonical_gt:
        return score
    return format_score


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------

def compute_score_em(
    solution_str: str,
    ground_truth: dict,
    method: str = "strict",
    format_score: float = 0.05,
    score: float = 1.0,
    extra_info: dict = None,
) -> float:
    """
    Combined accuracy + temporal + coverage reward.

    Args:
        solution_str: Full model rollout text
        ground_truth: {"target": ["Yes"]} or {"target": ["No"]}
        format_score: partial credit for structured but wrong answer
        score:        full credit for correct answer
        extra_info:   dict with:
          - end_date:        str  YYYY-MM-DD (market resolution date)
          - retrieved_dates: List[str|None]  pub dates of all retrieved docs
          - min_sources:     int  (default 5) — min acceptable sources used
          - max_sources:     int  (default 10) — max acceptable sources used

    Returns:
        float in [0, 1]
    """
    if extra_info is None:
        extra_info = {}

    r_accuracy = compute_accuracy_reward(solution_str, ground_truth, format_score, score)
    r_temporal = compute_temporal_reward(extra_info)
    r_coverage = compute_coverage_reward(extra_info)

    has_temporal = bool(extra_info.get("end_date") and extra_info.get("retrieved_dates"))
    has_coverage = bool(extra_info.get("retrieved_dates"))

    if not has_temporal and not has_coverage:
        # No retrieval data available — pure accuracy
        combined = r_accuracy
    elif not has_temporal:
        # Only coverage scoreable
        combined = (W_ACCURACY + W_TEMPORAL) * r_accuracy + W_COVERAGE * r_coverage
    elif not has_coverage:
        # Only temporal scoreable
        combined = (W_ACCURACY + W_COVERAGE) * r_accuracy + W_TEMPORAL * r_temporal
    else:
        # All three components
        combined = W_ACCURACY * r_accuracy + W_TEMPORAL * r_temporal + W_COVERAGE * r_coverage

    # Debug logging (1 in 64 calls)
    if random.randint(1, 64) == 1:
        targets = ground_truth.get("target", [])
        n_sources = len(extra_info.get("retrieved_dates", []))
        print("--------------------------------")
        print(f"Ground truth:    {targets}")
        print(f"Extracted:       {extract_solution(solution_str)!r}")
        print(f"r_accuracy:      {r_accuracy:.3f}  (w={W_ACCURACY})")
        print(f"r_temporal:      {r_temporal:.3f}  (w={W_TEMPORAL})")
        print(f"r_coverage:      {r_coverage:.3f}  (w={W_COVERAGE}, n_sources={n_sources})")
        print(f"combined:        {combined:.3f}")
        print(f"end_date:        {extra_info.get('end_date')}")
        print(f"source range:    [{extra_info.get('min_sources', DEFAULT_MIN_SOURCES)}, "
              f"{extra_info.get('max_sources', DEFAULT_MAX_SOURCES)}]")

    return combined


# Alias
compute_score_subem = compute_score_em
