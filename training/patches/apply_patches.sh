#!/usr/bin/env bash
# ============================================================
# Apply Polymarket training patches to a Search-R1 clone
# ============================================================
# Usage:
#   bash training/patches/apply_patches.sh <path-to-Search-R1>
#
# Example:
#   bash training/patches/apply_patches.sh /content/Search-R1
# ============================================================

set -e

SEARCHR1=${1:-"../Search-R1"}
REPO=$(dirname "$(dirname "$(realpath "$0")")")   # Research-Lookahead-AI root

echo "Patching Search-R1 at: $SEARCHR1"
echo "Using custom files from: $REPO"

# 1. Reward function
echo "[1/4] Installing polymarket reward function..."
cp "$REPO/training/reward/polymarket_reward.py" \
   "$SEARCHR1/verl/utils/reward_score/polymarket_em.py"

# Register in __init__.py if not already there
INIT="$SEARCHR1/verl/utils/reward_score/__init__.py"
if ! grep -q "polymarket_em" "$INIT"; then
    echo "from verl.utils.reward_score import polymarket_em" >> "$INIT"
    echo "  Added polymarket_em to reward_score __init__.py"
fi

# 2. Patched generation.py (passes end_date to retriever)
echo "[2/4] Patching generation.py..."
cp "$REPO/training/patches/generation.py" \
   "$SEARCHR1/search_r1/llm_agent/generation.py"

# 3. Patched main_ppo.py (registers polymarket reward + extra_info)
echo "[3/4] Patching main_ppo.py..."
cp "$REPO/training/patches/main_ppo.py" \
   "$SEARCHR1/verl/trainer/main_ppo.py"

# 4. Dataset + retriever server
echo "[4/4] Copying dataset and retriever..."
mkdir -p "$SEARCHR1/data"
# Processed parquet files (run process_polymarket.py first)
if [ -d "$REPO/training/data/polymarket_search" ]; then
    cp -r "$REPO/training/data/polymarket_search" "$SEARCHR1/data/polymarket_search"
    echo "  Copied polymarket_search parquet files"
else
    echo "  WARNING: training/data/polymarket_search not found"
    echo "  Run: python training/data/process_polymarket.py first"
fi

# Retriever server
cp "$REPO/training/retriever/exa_retriever_server.py" \
   "$SEARCHR1/exa_retriever_server.py"

echo ""
echo "All patches applied. Next steps:"
echo ""
echo "  1. Start retriever server:"
echo "     cd $SEARCHR1"
echo "     EXA_API_KEY=<your_key> python exa_retriever_server.py &"
echo ""
echo "  2. Run training:"
echo "     bash $REPO/training/configs/grpo_3b.sh"
echo ""
