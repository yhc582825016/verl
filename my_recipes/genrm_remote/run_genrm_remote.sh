# vllm server
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve verl-team/GenRM-CI-Test-1.5B --served_model_name genrm-demo

# sglang server
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang_router.launch_server --model-path verl-team/GenRM-CI-Test-1.5B --dp-size 4

set -x
export WANDB_API_KEY="04b01529fb630482bdf2f363456479f197ac5694"
export WANDB_MODE="online"
export WANDB_BASE_URL="https://api.bandw.top"
export OUTPUT_DIR=/mnt/code/yehangcheng/verl/outputs
export LOG_FILE=/mnt/code/yehangcheng/verl/run.log
export HYDRA_FULL_ERROR=1
# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

export CUDA_VISIBLE_DEVICES=4,5,6,7
nohup python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=['/mnt/code/yehangcheng/Evaluation/GEN/train_data/chatling-1204-1214-non_customized_df-v3-decuped.parquet','/mnt/code/yehangcheng/Evaluation/GEN/train_data/chatling-1204-1214-customized_df-v3-decuped.parquet'] \
    data.val_files=['/mnt/code/yehangcheng/Evaluation/GEN/test_data/chatling-1204-1214-non_customized_df-v3-decuped.parquet',/mnt/code/yehangcheng/Evaluation/GEN/test_data/chatling-1204-1214-customized_df-v3-decuped.parquet] \
    data.train_batch_size=256 \
    data.max_prompt_length=12000 \
    data.truncation='left' \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=/opt/users/ye/checkpoints/Qwen3-4B-Instruct-2507-general_total_425w_no_ace_reason_cat_online_datas_457w_shuffle_Math_rl_step_300_if_rl_step200_code_rl_step100 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=35000 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.n=8 \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path=/mnt/code/yehangcheng/verl/recipe/genrm_remote/reward_function_genrm.py \
    custom_reward_function.name=compute_score_batch \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='GenRM-qwen3-4B' \
    trainer.experiment_name='chatling-1204-1214-full_df-v3' \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.resume_mode='disable' \
    > /mnt/code/yehangcheng/verl/Qwen3-4B-Instruct-2507-general_total_425w_no_ace_reason_cat_online_datas_457w_shuffle_Math_rl_step_300_if_rl_step200_code_rl_step100_gen_rl.log 2>&1 &
