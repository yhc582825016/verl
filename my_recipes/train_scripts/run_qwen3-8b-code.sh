# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x
export HYDRA_FULL_ERROR=1
export WANDB_MODE="offline"
export RAY_EXPERIMENTAL_NCCL_GLOO_FALLBACK=0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

export SGLANG_ENGINE_ITERATION_TIMEOUT_S=1000000000
export SGLANG_ENGINE_REQUEST_TIMEOUT_S=1000000000

# export CUDA_LAUNCH_BLOCKING=1
export WANDB_KEY="04b01529fb630482bdf2f363456479f197ac5694"
export WANDB_BASE_URL="https://api.bandw.top"
export HF_DATASETS_CACHE='/opt/users/ye/HF_DATASETS_CACHE'
export CUDA_VISIBLE_DEVICES=4,5,6,7
TRAIN_FILE=/opt/users/ye/data/deepcoder/deepcoder_train.json
TEST_FILE=/opt/users/ye/data/deepcoder/test_livecodebench.json
# nohup 
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    trainer.val_before_train=True \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/opt/users/models/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=12000 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen3_8b_function_rm' \
    custom_reward_function.path=/mnt/code/yehangcheng/verl/recipe/rllm/rewards/rl_reward.py \
    custom_reward_function.name=rllm_reward_fn \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@ 
    # > /mnt/code/yehangcheng/verl/qwen3_8b_function_rm.log 2>&1 &