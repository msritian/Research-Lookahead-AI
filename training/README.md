# Polymarket RL Training — Full Pipeline

Train a `Qwen2.5-3B` base model to forecast binary prediction market outcomes using
**Search-R1's GRPO pipeline** with a live **Exa web search** backend, a composite
reward function, and temporal grounding to prevent future-data leakage.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Repository Layout](#2-repository-layout)
3. [End-to-End Data Flow](#3-end-to-end-data-flow)
4. [Component Deep-Dives](#4-component-deep-dives)
   - [Dataset Processing](#41-dataset-processing)
   - [Exa Retriever Server](#42-exa-retriever-server)
   - [Reward Function](#43-reward-function)
   - [generation.py patch](#44-generationpy-patch)
   - [main_ppo.py patch](#45-main_ppopy-patch)
   - [Training launch script](#46-training-launch-script)
5. [Reward Design](#5-reward-design)
6. [Temporal Grounding — How It Works](#6-temporal-grounding--how-it-works)
7. [Setup & Run Instructions](#7-setup--run-instructions)
8. [Key Hyperparameters](#8-key-hyperparameters)
9. [What Gets Trained vs What Is Fixed](#9-what-gets-trained-vs-what-is-fixed)
10. [Next Steps](#10-next-steps)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Research-Lookahead-AI                        │
│                                                                 │
│  training/                                                      │
│  ├── data/process_polymarket.py   ← CSV → parquet              │
│  ├── retriever/exa_retriever_server.py  ← live web search      │
│  ├── reward/polymarket_reward.py  ← composite reward           │
│  ├── patches/                                                   │
│  │   ├── generation.py   ← passes end_date to retriever        │
│  │   ├── main_ppo.py     ← registers reward + extra_info       │
│  │   └── apply_patches.sh ← one-command patcher                │
│  └── configs/grpo_3b.sh  ← GRPO training launch               │
└─────────────────────────────────────────────────────────────────┘
          ↓ apply_patches.sh
┌─────────────────────────────────────────────────────────────────┐
│                  Search-R1  (cloned separately)                 │
│                                                                 │
│  verl/trainer/main_ppo.py         ← PATCHED                    │
│  search_r1/llm_agent/generation.py ← PATCHED                   │
│  verl/utils/reward_score/polymarket_em.py  ← ADDED             │
│  data/polymarket_search/{train,test}.parquet ← COPIED          │
│  exa_retriever_server.py          ← COPIED                     │
└─────────────────────────────────────────────────────────────────┘
```

The **Search-R1** codebase is kept as a separate clone. We do not fork it.
All customisation lives here and is patched in with `apply_patches.sh`.

---

## 2. Repository Layout

```
training/
├── data/
│   └── process_polymarket.py      # converts CSV → veRL parquet format
│
├── retriever/
│   └── exa_retriever_server.py    # FastAPI /retrieve server backed by Exa
│
├── reward/
│   └── polymarket_reward.py       # accuracy + temporal + coverage reward
│
├── patches/
│   ├── generation.py              # patched Search-R1 rollout engine
│   ├── main_ppo.py                # patched veRL trainer entry-point
│   └── apply_patches.sh           # copies everything into Search-R1 clone
│
└── configs/
    └── grpo_3b.sh                 # GRPO training launch script (A100)
```

Root-level file used as input:

```
polymarket_dataset.csv             # ~200 binary Yes/No Polymarket questions
```

---

## 3. End-to-End Data Flow

```
polymarket_dataset.csv
        │
        ▼ process_polymarket.py
        │  • Filter non-binary rows (Up/Down/$X labels dropped)
        │  • Build Search-R1 instruction prompt per question
        │  • Store end_date (YYYY-MM-DD) in extra_info
        │  • 85/15 train/test split
        ▼
data/polymarket_search/
    ├── train.parquet
    └── test.parquet
        │
        ▼  apply_patches.sh copies into Search-R1/data/
        │
        ▼  grpo_3b.sh → python -m verl.trainer.main_ppo
                │
                │  For each training batch:
                │
                ▼ main_ppo.py (PATCHED)
                │  • Reads extra_info.end_date for every sample
                │  • Packs end_dates list into gen_batch.meta_info['end_dates']
                │  • Calls generation engine
                │
                ▼ generation.py (PATCHED) — multi-turn rollout loop
                │
                │  ┌──── for each search turn (up to max_turns=4) ─────┐
                │  │                                                    │
                │  │  1. Actor model generates text                     │
                │  │     until <search>…</search> or <answer>…</answer> │
                │  │                                                    │
                │  │  2. If <search> detected:                         │
                │  │     POST /retrieve                                 │
                │  │     {queries, topk, end_date=market.end_date}      │
                │  │            │                                       │
                │  │            ▼ exa_retriever_server.py               │
                │  │            │  • Calls Exa API                      │
                │  │            │  • Filters: published ≤ end_date      │
                │  │            │  • Returns docs + published_date      │
                │  │            ▼                                       │
                │  │     <information>…</information> appended          │
                │  │     published_dates collected → retrieved_dates    │
                │  │                                                    │
                │  │  3. Repeat for next turn                           │
                │  └────────────────────────────────────────────────────┘
                │
                │  After rollout completes:
                │  • retrieved_dates stored in meta_info['retrieved_dates']
                │  • Full rollout text passed to reward function
                │
                ▼ polymarket_reward.py (via main_ppo.py)
                │
                │  r_accuracy  (0.6 weight)
                │    → extracts last <answer>…</answer>
                │    → Yes/No match against ground_truth → 1.0 / 0.05 / 0.0
                │
                │  r_temporal  (0.2 weight)
                │    → fraction of retrieved_dates within [end_date - 365d, end_date]
                │    → penalises future articles AND stale articles
                │
                │  r_coverage  (0.2 weight)
                │    → n_sources = len(retrieved_dates)
                │    → 1.0 if n_sources in [5, 10], linear decay outside
                │
                │  combined = 0.6*r_acc + 0.2*r_temp + 0.2*r_cov
                │
                ▼
                GRPO advantage estimation (group of 4 rollouts per question)
                Actor gradient update (retrieved-doc tokens masked from loss)
                KL penalty vs reference model
```

---

## 4. Component Deep-Dives

### 4.1 Dataset Processing

**File:** `training/data/process_polymarket.py`

**Input:** `polymarket_dataset.csv` (~200 rows, columns: `market_id`, `question`,
`description`, `end_date`, `ground_truth_label`, `volume`, `slug`, ...)

**What it does:**

| Step | Detail |
|---|---|
| Filter | Drops rows where `ground_truth_label` ∉ {Yes, No} (e.g. Up/Down, $X ranges) |
| Prompt build | Wraps question in Search-R1's `<think>/<search>/<answer>` instruction |
| Resolution criteria | Appends `description` field (first 600 chars) as "Resolution criteria:" |
| Ground truth | Stores `["Yes"]` or `["No"]` in `reward_model.ground_truth.target` |
| `extra_info` | Stores `end_date` (YYYY-MM-DD), `market_id`, `question`, `split`, `index` |
| Split | 85% train / 15% test, shuffled (seed=42) |
| Output | `train.parquet` + `test.parquet` in veRL's expected schema |

**Schema of each parquet row:**

```json
{
  "data_source": "polymarket",
  "prompt": [{"role": "user", "content": "<instruction>\n\nQuestion: ...\n\nResolution criteria: ..."}],
  "ability": "fact-reasoning",
  "reward_model": {
    "style": "rule",
    "ground_truth": {"target": ["Yes"]}
  },
  "extra_info": {
    "split": "train",
    "index": 0,
    "market_id": "898677",
    "end_date": "2025-03-15",
    "question": "Will Trump sign the budget bill before March 2025?"
  }
}
```

**Run:**

```bash
python training/data/process_polymarket.py \
    --csv polymarket_dataset.csv \
    --out training/data/polymarket_search \
    --val-ratio 0.15 \
    --seed 42
```

---

### 4.2 Exa Retriever Server

**File:** `training/retriever/exa_retriever_server.py`

A **FastAPI microservice** that exposes the same `/retrieve` endpoint contract as
Search-R1's built-in corpus-based retriever, but routes queries to the **live Exa
web search API** instead.

**Why a server instead of calling Exa directly?**

Search-R1's rollout engine expects a synchronous HTTP POST at a fixed URL. The server
is the drop-in replacement that lets us swap the backend without touching the core
rollout loop.

**Key additions over a plain Exa wrapper:**

| Feature | Detail |
|---|---|
| `end_date` param | If provided, passes `end_published_date` to Exa so only pre-resolution articles are returned |
| `published_date` in response | Each `document` dict includes `"published_date": "YYYY-MM-DD"` so the rollout engine can collect dates for the reward |
| Graceful error handling | Any Exa exception returns an empty list (training continues) |
| `None` score fallback | If Exa doesn't return a relevance score, assigns `1.0 - i*0.05` as a rank-based proxy |

**Request / Response:**

```
POST /retrieve
{
  "queries":       ["Will X happen?"],
  "topk":          3,
  "return_scores": true,
  "end_date":      "2025-03-15"   ← optional, YYYY-MM-DD
}

→ {
  "result": [[
    {
      "document": {
        "contents":       "\"Article Title\"\narticle text up to 2000 chars",
        "published_date": "2025-02-20"
      },
      "score": 0.94
    },
    ...
  ]]
}
```

**Run:**

```bash
export EXA_API_KEY=your_key_here
python training/retriever/exa_retriever_server.py
# Health check:
curl http://127.0.0.1:8000/health
```

Port is configurable via `RETRIEVER_PORT` env var (default: 8000).

---

### 4.3 Reward Function

**File:** `training/reward/polymarket_reward.py`
**Installed to:** `Search-R1/verl/utils/reward_score/polymarket_em.py`

Three-component composite reward returned as a single scalar in `[0, 1]`:

```
reward = 0.6 × r_accuracy + 0.2 × r_temporal + 0.2 × r_coverage
```

See [Section 5](#5-reward-design) for full details.

---

### 4.4 generation.py patch

**File:** `training/patches/generation.py`
**Replaces:** `Search-R1/search_r1/llm_agent/generation.py`

This is the **rollout engine** — the core multi-turn loop where the model generates
text, detects `<search>` tags, queries the retriever, and injects results back.

**Changes over original:**

| Change | Purpose |
|---|---|
| `GenerationConfig.end_dates: List[str\|None]` | New field, one entry per sample in the batch |
| `_sample_end_dates` stored on manager | Per-sample market end dates, loaded from `meta_info['end_dates']` at rollout start |
| `_retrieved_dates_per_sample` collected | Accumulates `published_date` from each retrieved doc, per sample, across all turns |
| `_batch_search()` receives `sample_indices` | Knows which samples triggered this batch of searches |
| `end_date` forwarded in POST payload | Sent to retriever server so Exa filters results to pre-resolution articles only |
| `retrieved_dates` written to `meta_info` | At rollout end, stored so `main_ppo.py` can pass them to the reward function |

**Temporal cutoff strategy:** When a batch of searches arrives (potentially from
multiple samples), the engine takes the **earliest** `end_date` across those samples
as the cutoff. This is conservative — it means no sample in the batch can receive
articles from after its market resolved.

---

### 4.5 main_ppo.py patch

**File:** `training/patches/main_ppo.py`
**Replaces:** `Search-R1/verl/trainer/main_ppo.py`

The **trainer entry-point** that orchestrates data loading, rollout, reward, and
gradient updates via veRL.

**Changes over original:**

| Change | Purpose |
|---|---|
| `import polymarket_em` | Registers the new reward module |
| `_select_rm_score_fn('polymarket')` | Routes Polymarket samples to `polymarket_em.compute_score_em` |
| `extra_info` built per sample | Reads `data_item.non_tensor_batch['extra_info']` (contains `end_date`) |
| `retrieved_dates` injected into `extra_info` | Read from `data.meta_info['retrieved_dates']` populated by `generation.py` |
| Full `extra_info` passed to reward | Reward function receives both `end_date` and `retrieved_dates` |

**Reward call signature:**

```python
score = polymarket_em.compute_score_em(
    solution_str=sequences_str,    # full model rollout text
    ground_truth=ground_truth,     # {"target": ["Yes"]} or {"target": ["No"]}
    format_score=0.05,
    extra_info={
        "end_date":        "2025-03-15",
        "retrieved_dates": ["2025-02-01", "2025-03-10", None, "2025-01-15"],
        "market_id":       "898677",
        "question":        "Will Trump sign...",
        ...
    }
)
```

---

### 4.6 Training launch script

**File:** `training/configs/grpo_3b.sh`

Launches `verl.trainer.main_ppo` with all hyperparameters tuned for a **single A100
40GB GPU**.

**Key settings:**

| Setting | Value | Reason |
|---|---|---|
| `BASE_MODEL` | `Qwen/Qwen2.5-3B` | Base (not instruct) — blank slate for RL |
| `algorithm.adv_estimator` | `grpo` | Group Relative Policy Optimisation |
| `actor_rollout_ref.rollout.n_agent` | `4` | 4 rollouts per question for advantage estimation |
| `max_turns` | `4` | Up to 4 search rounds per question |
| `retriever.topk` | `3` | 3 docs per search query |
| `state_masking` | `true` | Retrieved-doc tokens excluded from actor loss |
| FSDP offload | `param + grad + optimizer` | Fits 3B model + optimizer on 40GB |
| `kl_loss_coef` | `0.001` | Soft KL penalty vs reference model |

**Reward environment variables** (set in the script, overridable):

```bash
REWARD_MIN_SOURCES=5      # coverage reward: min acceptable sources
REWARD_MAX_SOURCES=10     # coverage reward: max acceptable sources
REWARD_MAX_PAST_DAYS=365  # temporal reward: articles older than this are penalised
```

---

## 5. Reward Design

### r_accuracy (weight = 0.6)

Checks whether the model's final `<answer>` block matches the ground truth.

```
Extract last <answer>…</answer> from rollout text
Normalise → canonical "Yes" or "No" (case-insensitive, strips punctuation)
```

| Condition | Score |
|---|---|
| No `<answer>` tag at all | 0.0 |
| Tag present, but answer is not parseable as Yes/No | 0.05 (format credit) |
| Parseable but wrong (e.g. "Yes" when truth is "No") | 0.05 |
| Correct Yes or No | 1.0 |

### r_temporal (weight = 0.2)

Checks that retrieved articles were published in the valid window:

```
valid window = [end_date − max_past_days,  end_date]
               (default: last 365 days before market resolved)

score = fraction of retrieved articles with a valid published_date
```

| Article | Treatment |
|---|---|
| Published after `end_date` | Invalid (future leakage) |
| Published before `end_date − max_past_days` | Invalid (too stale) |
| Within the window | Valid |
| `published_date` is None | Excluded from fraction |

Returns 1.0 if no `end_date` or no `retrieved_dates` are available (no penalty during
early training before retrieval is working).

### r_coverage (weight = 0.2)

Rewards the model for using an appropriate breadth of sources.

```
n_sources = total documents retrieved across all search turns

score = 1.0                          if min_sources ≤ n_sources ≤ max_sources
      = n_sources / min_sources      if n_sources < min_sources   (too few, linear)
      = max_sources / n_sources      if n_sources > max_sources   (too many, linear)
```

Default range: `[5, 10]` sources. Configurable via `REWARD_MIN_SOURCES` /
`REWARD_MAX_SOURCES` env vars.

### Graceful degradation

If only some components are available, the weights are redistributed:

| Available components | Formula |
|---|---|
| All three | `0.6·acc + 0.2·temp + 0.2·cov` |
| accuracy + coverage only | `0.8·acc + 0.2·cov` |
| accuracy + temporal only | `0.8·acc + 0.2·temp` |
| accuracy only | `1.0·acc` |

---

## 6. Temporal Grounding — How It Works

Temporal grounding is the mechanism that prevents the model from "cheating" by
reading articles published after the market's resolution date.

It operates at **two levels**:

### Level 1 — Retrieval-time filtering (hard cutoff)

When `generation.py` calls the retriever server, it passes `end_date` in the POST
body. The server forwards this to Exa's `end_published_date` parameter. Exa only
returns articles published **before** that date.

This means the model literally cannot see future information during a training rollout.

### Level 2 — Reward-time penalisation (soft signal)

Even with the hard cutoff, some articles may be too stale to be useful. The
`r_temporal` reward penalises any retrieved article published outside the valid window.
This teaches the model to search for **recent but pre-resolution** articles rather
than just any articles that technically pass the hard cutoff.

### How end_date flows through the system

```
CSV (end_date column)
    ↓ process_polymarket.py
parquet extra_info.end_date
    ↓ veRL data loader
data_item.non_tensor_batch['extra_info']['end_date']
    ↓ main_ppo.py (PATCHED)
gen_batch.meta_info['end_dates']  (List, one per sample in batch)
    ↓ generation.py (PATCHED)
_sample_end_dates[sample_idx]
    ↓ _batch_search() → POST /retrieve {"end_date": "..."}
    ↓ exa_retriever_server.py → Exa API end_published_date
    ↓ response.document.published_date (per doc)
_retrieved_dates_per_sample[sample_idx]  (accumulated across all turns)
    ↓ meta_info['retrieved_dates']
    ↓ main_ppo.py → extra_info['retrieved_dates']
    ↓ polymarket_reward.py → r_temporal
```

---

## 7. Setup & Run Instructions

### Prerequisites

- Google Colab Pro (A100 40GB) or equivalent
- Python 3.10+
- An Exa API key
- A WandB account (for logging; set `WANDB_API_KEY`)

### Step 1 — Clone Search-R1

```bash
git clone https://github.com/PeterGriffinJin/Search-R1
cd Search-R1 && pip install -e . && cd ..
```

### Step 2 — Install extra dependencies

```bash
pip install exa_py fastapi uvicorn pandas pyarrow
```

### Step 3 — Process the dataset

Run from the `Research-Lookahead-AI` root:

```bash
python training/data/process_polymarket.py \
    --csv polymarket_dataset.csv \
    --out training/data/polymarket_search
```

Expected output:
```
Loading polymarket_dataset.csv...
  Loaded 193 binary markets (7 non-binary dropped)
  Train: 164  |  Val: 29
  train: Yes=82  No=82
  val:   Yes=14  No=15
  Saved 164 records → training/data/polymarket_search/train.parquet
  Saved 29 records → training/data/polymarket_search/test.parquet
```

### Step 4 — Apply patches to Search-R1

```bash
bash training/patches/apply_patches.sh /path/to/Search-R1
```

This copies:
- `polymarket_reward.py` → `Search-R1/verl/utils/reward_score/polymarket_em.py`
- `patches/generation.py` → `Search-R1/search_r1/llm_agent/generation.py`
- `patches/main_ppo.py` → `Search-R1/verl/trainer/main_ppo.py`
- `retriever/exa_retriever_server.py` → `Search-R1/exa_retriever_server.py`
- `data/polymarket_search/` → `Search-R1/data/polymarket_search/`

### Step 5 — Start the Exa retriever server

In a separate terminal (or background process):

```bash
export EXA_API_KEY=your_key_here
cd Search-R1
python exa_retriever_server.py &

# Verify it's running:
curl http://127.0.0.1:8000/health
# → {"status": "ok", "backend": "exa"}
```

### Step 6 — Launch training

```bash
cd Search-R1
mkdir -p logs verl_checkpoints
export WANDB_API_KEY=your_wandb_key

bash /path/to/Research-Lookahead-AI/training/configs/grpo_3b.sh
```

Logs are written to `Search-R1/logs/polymarket-search-r1-grpo-qwen2.5-3b-em.log`.
Checkpoints saved every 50 steps to `Search-R1/verl_checkpoints/`.

---

## 8. Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Base model | `Qwen/Qwen2.5-3B` | Base, not instruct |
| Algorithm | GRPO | Group Relative Policy Optimisation |
| Rollouts per question | 4 (`n_agent=4`) | Group size for advantage estimation |
| Max search turns | 4 | Model can issue up to 4 `<search>` queries |
| Docs per search | 3 (`topk=3`) | Articles fetched per search query |
| Max response length | 500 tokens | Keeps rollouts manageable on 40GB |
| Max prompt length | 4096 tokens | Includes instruction + retrieved context |
| Learning rate | 1e-6 | Low to preserve pretrained knowledge |
| KL coefficient | 0.001 | Soft constraint vs reference model |
| State masking | enabled | Retrieved doc tokens excluded from actor loss |
| Train batch size | 64 | Global batch across all rollouts |
| Epochs | 10 | Full passes over training data |
| Save/eval frequency | 50 steps | |
| Reward: accuracy weight | 0.6 | Primary signal |
| Reward: temporal weight | 0.2 | Penalise stale/future sources |
| Reward: coverage weight | 0.2 | Penalise too few/many sources |
| Coverage range | 5–10 sources | Configurable via env vars |
| Max source age | 365 days | Configurable via `REWARD_MAX_PAST_DAYS` |

---

## 9. What Gets Trained vs What Is Fixed

| Component | Status | Notes |
|---|---|---|
| `Qwen2.5-3B` actor weights | **Trained** | Core model being fine-tuned |
| Reference model | **Fixed** | KL anchor, not updated |
| Exa retriever | **Fixed** | External API, not a learned component |
| Retriever server | **Fixed** | Just a routing layer |
| Reward function | **Fixed** | Rule-based, not a neural reward model |

The training signal flows entirely through the **accuracy / temporal / coverage
rewards** back into the actor via GRPO advantage estimation. The model learns:

1. **When to search** — issue `<search>` queries at the right moments
2. **What to search for** — queries that return relevant, temporally valid articles
3. **How to reason** — use retrieved evidence to arrive at a correct Yes/No answer
4. **Answer format** — always wrap final answer in `<answer>...</answer>`

---

## 10. Next Steps

| Priority | Task |
|---|---|
| High | Evaluate trained checkpoint on held-out test set (29 markets) |
| High | Compare against Search-R1's pretrained checkpoint on same test set |
| Medium | Increase dataset size — scrape more binary Polymarket questions |
| Medium | Tune `n_agent` to 8 for better advantage estimation (needs multi-GPU) |
| Medium | Add faithfulness reward — NLI check that answer is grounded in retrieved docs |
| Low | Curriculum: start with EM-only reward, add temporal/coverage after convergence |
| Low | Experiment with `Qwen2.5-7B` base for higher capacity |
