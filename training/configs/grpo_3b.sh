#!/usr/bin/env bash
# ============================================================
# Polymarket GRPO Training — Qwen2.5-3B base, single A100
# ============================================================
# Run from the Search-R1 repo root after:
#   1. pip install -e .   (inside Search-R1/)
#   2. Copy custom files  (see training/README.md)
#   3. Process dataset    (python training/data/process_polymarket.py)
#   4. Start retriever    (python training/retriever/exa_retriever_server.py &)
# ============================================================

set -e

export CUDA_VISIBLE_DEVICES=0          # single A100
export DATA_DIR=data/polymarket_search  # output of process_polymarket.py
export BASE_MODEL=Qwen/Qwen2.5-3B      # base model (not instruct, not checkpoint)
export EXPERIMENT_NAME=polymarket-search-r1-grpo-qwen2.5-3b-em

# Coverage reward: acceptable range of web sources per question
# Model is penalised if it uses fewer than MIN or more than MAX sources
export REWARD_MIN_SOURCES=${REWARD_MIN_SOURCES:-5}
export REWARD_MAX_SOURCES=${REWARD_MAX_SOURCES:-10}
# Temporal reward: sources older than this many days before end_date are penalised
export REWARD_MAX_PAST_DAYS=${REWARD_MAX_PAST_DAYS:-365}

export WANDB_PROJECT="Polymarket-SearchR1"
export VLLM_ATTENTION_BACKEND=XFORMERS  # required for Qwen2.5 + vLLM

# Adjust for single A100 (40GB):
#   - Smaller batch sizes than the 8-GPU default
#   - FSDP offloading enabled to fit model + optimizer
#   - max_response_length=500 keeps rollouts manageable

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n_agent=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    algorithm.no_think_rl=false \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=10 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee logs/$EXPERIMENT_NAME.log
