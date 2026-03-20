# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x
export HYDRA_FULL_ERROR=1
export WANDB_MODE="online"
export RAY_EXPERIMENTAL_NCCL_GLOO_FALLBACK=0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# export SGLANG_ENGINE_ITERATION_TIMEOUT_S=1000000000
# export SGLANG_ENGINE_REQUEST_TIMEOUT_S=1000000000
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export PYTHONPATH="/mnt/code/yehangcheng/verl/recipe:/mnt/code/yehangcheng/verl/recipe:${PYTHONPATH:-}"

# export CUDA_LAUNCH_BLOCKING=1
export WANDB_KEY="04b01529fb630482bdf2f363456479f197ac5694"
export WANDB_BASE_URL="https://api.bandw.top"
export HF_DATASETS_CACHE='/opt/users/ye/HF_DATASETS_CACHE'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 已保存 primeintellect，共 16252 行 → /opt/users/ye/data/deepcoder/split/deepcoder_train_modified_primeintellect.parquet
# 已保存 taco，共 7436 行 → /opt/users/ye/data/deepcoder/split/deepcoder_train_modified_taco.parquet
# 已保存 livecodebench，共 599 行 → /opt/users/ye/data/deepcoder/split/deepcoder_train_modified_livecodebench.parquet
project_name=DAPO
exp_name=DAPO_Qwen3_8B_Base_sft_202w_concat_lmsys_278w_24k_math_step_480_IF_step_600_code_r1_data
CKPTS_DIR=/mnt/code/yehangcheng/verl/checkpoints/${project_name}/${exp_name}
TRAIN_FILE=/opt/users/ye/data/deepcoder/split/deepcoder_train_modified_taco.parquet
TEST_FILE=/opt/users/ye/data/deepcoder/test_livecodebench.parquet
nohup python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    trainer.val_before_train=True \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=24000 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path=/opt/users/ye/checkpoints/DAPO-Qwen3_8B_Base_sft_202w_concat_lmsys_278w_24k_math_step_480_IF_step_600 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=32000 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    reward_model.reward_manager=dapo_batch \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    custom_reward_function.path=/mnt/code/yehangcheng/verl/recipe/code_r1_reward/code_r1_compute.py \
    custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.total_epochs=10 $@ > /mnt/code/yehangcheng/verl/Qwen3_8B_Base_sft_202w_concat_lmsys_278w_24k_math_step_480_IF_step_600_code_r1_data.log 2>&1 &