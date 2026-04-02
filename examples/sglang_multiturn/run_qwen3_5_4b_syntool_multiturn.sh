#!/usr/bin/env bash

set -xeuo pipefail
cd /mnt/code/yehangcheng/verl
export WANDB_MODE="online"
export WANDB_BASE_URL="https://api.bandw.top"
export OUTPUT_DIR=/mnt/code/yehangcheng/cache_file/verl/outputs
export LOG_FILE=/mnt/code/yehangcheng/cache_file/verl/run.log
export HYDRA_FULL_ERROR=1
export HYDRA_RUN_DIR=/mnt/code/yehangcheng/cache_file/verl/outputs/ppo_run
export WANDB_API_KEY="04b01529fb630482bdf2f363456479f197ac5694"
export CUDA_VISIBLE_DEVICES=4,5,6,7
mkdir -p $HYDRA_RUN_DIR
mkdir -p $OUTPUT_DIR

ulimit -n 65535

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/opt/users/models/Qwen3-4B}"
TRAIN_FILE="${TRAIN_FILE:-/mnt/code/yehangcheng/github/verl/examples/sglang_multiturn/syntool_recall_filter/train_filtered.parquet}"
VAL_FILE="${VAL_FILE:-/mnt/code/yehangcheng/github/verl/examples/sglang_multiturn/syntool_recall_filter/test_filtered.parquet}"
PROJECT_NAME="${PROJECT_NAME:-sglang-syntool-multiturn}"
EXP_NAME="${EXP_NAME:-qwen3_5_4b_syntool_multiturn}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-10240}"
ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))}"
INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 3))}"
CKPTS_DIR="${CKPTS_DIR:-/mnt/code/yehangcheng/checkpoint/Agent_model/${PROJECT_NAME}/${EXP_NAME}}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-0}"

nohup python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.return_raw_chat=True \
    data.train_batch_size=32 \
    data.val_batch_size=8 \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.custom_cls.path="${PROJECT_DIR}/my_recipes/syntool/dataset.py" \
    data.custom_cls.name=SyntoolRLHFDataset \
    reward.custom_reward_function.path="${PROJECT_DIR}/my_recipes/syntool/reward.py" \
    reward.custom_reward_function.name=compute_syntool_score \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${INFER_PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${INFER_PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=32 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=32 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=null \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_DIR}/my_recipes/syntool/agent_loop_config.yaml" \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    trainer.logger="['console','wandb']" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.n_gpus_per_node=4 \
    trainer.log_val_generations="${LOG_VAL_GENERATIONS}" \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    "$@" > /mnt/code/yehangcheng/logs/qwen35_4b_verl.log 2>&1 &
