set -x
export WANDB_MODE="online"
export WANDB_BASE_URL="https://api.bandw.top"
export OUTPUT_DIR=/mnt/code/yehangcheng/verl/outputs
export LOG_FILE=/mnt/code/yehangcheng/verl/run.log
export HYDRA_FULL_ERROR=1
export HYDRA_RUN_DIR=/mnt/code/yehangcheng/verl/outputs/ppo_run
export WANDB_API_KEY="04b01529fb630482bdf2f363456479f197ac5694"
mkdir -p $HYDRA_RUN_DIR
mkdir -p $OUTPUT_DIR

project_name='verl_if_grpo'
experiment_name='if_grpo_test'
save_path=/mnt/code/yehangcheng/verl/checkpoints/${project_name}/${experiment_name}
# https://api.bandw.top
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH=/mnt/code/yehangcheng/verl/recipe/instruct_following
# cd /mnt/code/yehangcheng/verl/recipe/insturct_following
HOME=/mnt/code/yehangcheng/verl/recipe/insturct_following
# nohup 
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/code/yehangcheng/verl/recipe/insturct_following/rl_dataset_train.parquet \
    data.val_files=/mnt/code/yehangcheng/verl/recipe/insturct_following/ifeval_test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    actor_rollout_ref.model.path=/opt/users/models/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.max_critic_ckpt_to_keep=3 \
    trainer.default_local_dir=$save_path \
    custom_reward_function.path=/mnt/code/yehangcheng/verl/recipe/insturct_following/reward_function.py \
    custom_reward_function.name=instruction_following_reward_function \
    # > /mnt/code/yehangcheng/verl/recipe/insturct_following/Qwen3-4B-Instruct-2507-instruction_following.log 2>&1 &