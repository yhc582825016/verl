# Multi-Turn Rollout Example (GSM8K)

This example demonstrates how to perform **multi-turn rollout** using SGLang with a tool-calling capable model (e.g., Qwen2.5-3B) on the GSM8K dataset.

## Usage

### Step 1: Download GSM8K Dataset

```bash
cd examples/data_preprocess
python3 gsm8k_multiturn_w_tool.py
```

This will download and preprocess the GSM8K dataset into ~/data/gsm8k/.

### Step 2: Run Multi-Turn Rollout

If you have 8 GPUs
Use the standard 8-GPU script:

```bash
cd your_verl_root_dir
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
```

If you have only 4 GPUs
Use the fallback 4-GPU script:

```bash
cd your_verl_root_dir
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn_4xgpu.sh 
```

## Notes

- The rollout supports multi-turn conversations with tool-calling capabilities.
- Current tools are used for GSM8K answer evaluation.
- Future versions may extend to search and code interpreter tools.

## Gym Reward Environment

verl now supports gym-style reward environments through interaction config:

- Use interaction class `verl.interactions.gym_interaction.GymInteraction`
- Use env name `nemo_gym_env` from `verl.interactions.gym_env`
- Example config: `examples/sglang_multiturn/config/interaction_config/gym_interaction_config.yaml`

For each training sample, set `extra_info.interaction_kwargs` with at least:

- `name: "gym"`
- `env_config`: for example:
  - `name: "nemo_gym_env"`
  - `verify_url: "http://127.0.0.1:18001/verify"`
  - `prompt_key`, `reward_key`, `done_on_verify` (optional)

When `GymInteraction` returns per-turn reward, verl writes it into `rm_scores`
directly, so PPO/GRPO can train with gym-provided reward without a custom
reward function.
