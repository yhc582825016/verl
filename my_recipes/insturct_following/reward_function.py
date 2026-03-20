# coding: utf-8
"""
Instruction-following reward function (strict, averaged).
- 所有判定信息从 extra_info 里取
- 奖励 = 命中比例 = (#True) / (总指令数)
- 仅返回一个数字(float)，例如 0.6666
"""

from typing import Dict, List, Any, Optional
import os, sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
import instructions_registry
import json

def _strict_hits(
                 solution_str: str,
                 instruction_id_list: List[str],
                 kwargs_list: List[Dict[str, Any]]) -> List[bool]:
    """返回逐指令命中列表 [True/False,...]，不提前返回。"""
    assert len(instruction_id_list) == len(kwargs_list), "指令数与kwargs数不一致"

    hits: List[bool] = []
    for idx, inst_id in enumerate(instruction_id_list):
        inst_cls = instructions_registry.INSTRUCTION_DICT[inst_id]
        inst = inst_cls(inst_id)
        # print("idx",idx)
        # print("kwargs_list",kwargs_list)
        inst.build_description(**(kwargs_list[idx] or {}))
        # args = inst.get_instruction_args()
        # if args and "prompt" in args:
        #     inst.build_description(prompt=prompt)

        ok = bool(solution_str.strip()) and bool(inst.check_following(solution_str))
        hits.append(ok)
    return hits


def instruction_following_reward_function(
    data_source: Optional[Any],
    solution_str: str,
    ground_truth: Optional[Any] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Args:
        data_source: 保留接口，不使用
        solution_str: 待评测的模型输出
        ground_truth: 保留接口，不使用
        extra_info: 必含:
            - "prompt": str
            - "instruction_id_list": List[str]
            - "kwargs": List[Dict]
          可选:
            - "round_digits": int  # 结果四舍五入的小数位

    Returns:
        float: 命中比例（例如 0.6666）
    """
    # try:
    # extra_info = json.loads(extra_info)
    # print(extra_info)
    instruction_id_list = extra_info["instruction_id_list"]
    kwargs_list = extra_info["instruction_kwargs"]
    round_digits = extra_info.get("round_digits", 4)
    
    hits = _strict_hits( solution_str, instruction_id_list, kwargs_list)
    print({"solution_str":solution_str,"instruction_id_list":instruction_id_list,"kwargs_list":kwargs_list,"hits":hits})
    print("\n")
    if not hits:
        return 0.0  # 没有指令则给0
    if all(hits):
        return 1.0
    else:
        return 0.0
    # score = sum(1 for h in hits if h) / float(len(hits))
    # if isinstance(round_digits, int) and round_digits >= 0:
    #     score = round(score, round_digits)
    # return score
    # except Exception as e:
    #     print(f"[WARN][reward_function] {e}")
    #     return 0.0

# extra_info = {
#     "prompt": "Write a 300+ word summary of ...",
#     "instruction_id_list": [
#         "punctuation:no_comma",
#         "detectable_format:number_highlighted_sections",
#         "length_constraints:number_words"
#     ],
#     "instruction_kwargs": [{}, {"num_highlights": 3}, {"relation": "at least", "num_words": 300}],
#     "round_digits": 4  # 想得到 0.6666 这种固定小数位就加这个
# }

# solution_str =" Raymond III was a prominent figure in the Crusader states serving as the Count of Tripoli from 1152 until his death in 1187. Born to Hodierna of Jerusalem and Raymond II Count of Tripoli he ascended to power after his father's death. Raymond played significant roles in the politics and conflicts of the Outremer distinguished by his alliances and disputes with other key figures.\n\n*highlighted section part 1*\n### Political Alliances and Conflicts\nRaymond's political career was marked by strategic alliances and conflicts notably with the Knights Templar and the Knights Hospitaller. He initially supported the Knights Templar against Nur ad-Din but later turned against them aligning with the Hospitallers. His relations with the Kingdom of Jerusalem were complex particularly with Kings Amalric I and Baldwin IV with whom he had periodic tensions and reconciliations.\n\n*highlighted section part 2*\n### Military Campaigns\nRaymond participated in several military campaigns including the disastrous invasion of Egypt in 1168 and the Battle of Montgisard in 1177 where the Crusader forces secured a significant victory. His military strategies often involved defensive measures rather than aggressive campaigns. He was instrumental in negotiating truces with Saladin which provided temporary peace in the region.\n\n*highlighted section part 3*\n### Role in the Fall of Jerusalem\nRaymond's role in the events leading to the fall of Jerusalem is noteworthy. Despite his efforts to maintain peace his actions were often seen as controversial particularly his alliance with Saladin against Guy of Lusignan. This alliance was a contributing factor to the disunity among the Crusader states which ultimately weakened their defenses. Raymond met his end during the Battle of Hattin in 1187 where he was captured and later executed by Saladin.\n\nRaymond III's legacy is complex influenced by his attempts to navigate the treacherous political landscape of the Crusader states. His actions both in alliance and opposition to key figures and powers shaped the history of the region during a turbulent era."
# reward = instruction_following_reward_function(None, solution_str, extra_info=extra_info)
# print(reward)
# # 仅返回一个数字，例如 0.6667
