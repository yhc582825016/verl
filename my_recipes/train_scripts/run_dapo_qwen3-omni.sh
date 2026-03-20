#!/usr/bin/env bash
set -xeuo pipefail

# This script targets the patched verl under github/verl.
cd /mnt/code/yehangcheng/github/verl

export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/code/yehangcheng/verl/outputs}"
export LOG_DIR="${LOG_DIR:-/mnt/code/yehangcheng/verl/logs}"

project_name="${PROJECT_NAME:-Qwen3-Omni-GRPO}"
exp_name="${EXP_NAME:-qwen3_omni_30b_gspo_if_306_math_grpo}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"

MODEL_PATH="${MODEL_PATH:-/opt/users/ye/checkpoints/Qwen3-Omni-30B-A3B-Instruct-gspo-if-306-lora/checkpoint-250-merged}"
TRAIN_FILE="${TRAIN_FILE:-/mnt/code/yehangcheng/sft_data/DAPO-Math-17k/data/dapo-math-17k-dup.parquet}"
TEST_FILE="${TEST_FILE:-/mnt/code/yehangcheng/sft_data/AIME-2024/data/aime-2024-dup.parquet}"
CKPTS_DIR="${CKPTS_DIR:-/mnt/code/yehangcheng/verl/checkpoints/${project_name}/${exp_name}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${exp_name}.log}"

adv_estimator="${ADV_ESTIMATOR:-grpo}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-12000}"
train_prompt_bsz="${TRAIN_PROMPT_BSZ:-128}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-8}"
train_prompt_mini_bsz="${TRAIN_PROMPT_MINI_BSZ:-4}"

temperature="${TEMPERATURE:-1.0}"
top_p="${TOP_P:-1.0}"
top_k="${TOP_K:--1}"
val_top_p="${VAL_TOP_P:-0.7}"

sp_size="${SP_SIZE:-4}"
gen_tp="${GEN_TP:-4}"
fsdp_size="${FSDP_SIZE:-8}"
offload="${OFFLOAD:-True}"
use_dynamic_bsz="${USE_DYNAMIC_BSZ:-True}"

actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${CKPTS_DIR}"

# Expected parquet/jsonl schema:
# - `messages` or `prompt`
# - optional `audios`, `images`, `videos`
# - optional `mm_processor_kwargs`, e.g. {"use_audio_in_video": false}
#
# Example multimodal row:
# {
#   "messages": [{"role": "user", "content": "<audio><image><video>Describe all modalities."}],
#   "audios": ["file:///path/to/a.wav"],
#   "images": ["file:///path/to/b.png"],
#   "videos": [{"video": "file:///path/to/c.mp4"}],
#   "mm_processor_kwargs": {"use_audio_in_video": false},
#   "reward_model": {"style": "rule", "ground_truth": ["..."]}
# }
#
# Current default TRAIN_FILE / TEST_FILE are text-only math datasets.
# This means the script will train Qwen3-Omni as a text RL model first,
# without exercising its audio / image / video branches.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

nohup python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="${adv_estimator}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.messages_key=messages \
    data.audio_key=audios \
    data.image_key=images \
    data.video_key=videos \
    data.mm_processor_kwargs_key=mm_processor_kwargs \
    data.truncation='left' \
    data.filter_overlong_prompts=True \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.train_batch_size="${train_prompt_bsz}" \
    actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${gen_tp}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature="${temperature}" \
    actor_rollout_ref.rollout.top_p="${top_p}" \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${temperature}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${val_top_p}" \
    actor_rollout_ref.rollout.val_kwargs.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.enable_audio_output=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${train_prompt_mini_bsz}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_ppo_max_token_len}" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${fsdp_size}" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${offload}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${offload}" \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${sp_size}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
    actor_rollout_ref.ref.fsdp_config.param_offload="${offload}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${sp_size}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=20 \
    trainer.total_epochs=5 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.log_val_generations=4 \
    "$@" > "${LOG_FILE}" 2>&1 &
