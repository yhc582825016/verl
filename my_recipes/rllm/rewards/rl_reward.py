import os, sys
sys.path.append('/mnt/code/yehangcheng/verl/recipe/rllm')
sys.path.append('/code/yehangcheng/verl/recipe/rllm')
from rewards.reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from rewards.math_reward import RewardMathFn
from rewards.code_reward import rllm_reward_fn_code 
from rewards.math_reward import rllm_reward_fn_math
from typing import Union, List
import json 

# Check RewardConfig to understand the config values.
class RLRewardFn(RewardFn):
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.math_reward_fn = RewardMathFn(config)
        self.cot_reward_fn = None

    def __call__(self, input: RewardInput) -> RewardOutput:
        reward_type = input.problem_type
        reward = 0
        is_correct = False
        if reward_type == RewardType.MATH:
            math_reward_output = self.math_reward_fn(input)
            reward += self.config.math_reward_weight * math_reward_output.reward
            is_correct = math_reward_output.is_correct
        elif reward_type == RewardType.CODE:
            pass
        else:
            raise ValueError(f"Unsupported reward type: {reward_type}")
        
        if self.config.cot_reward_weight != 0:
            cot_reward_output = self.cot_reward_fn(input)
            reward += self.config.cot_reward_weight * cot_reward_output.reward
        
        return RewardOutput(
            reward=reward,
            is_correct=is_correct
        )

def rllm_reward_fn(data_source: str, solution_str: str, ground_truth: Union[str, List[str]], extra_info={}, **kwargs):
    if data_source in ["apps", "taco", "code_contests", "codeforces", "livecodebench", "kodcode", "leetcode", "primeintellect", "humanevalplus"]:
        # print('solution_str',solution_str)
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return False 
        return rllm_reward_fn_code(data_source, solution_str, ground_truth, **kwargs)
    else:
        return rllm_reward_fn_math(data_source, solution_str, ground_truth, extra_info, **kwargs)