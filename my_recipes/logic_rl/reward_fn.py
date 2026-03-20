import re
from typing import Dict, Tuple, Optional
from typing import List, Any
def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """从模型输出中抽取 <answer>...</answer> 的内容。

    Args:
        solution_str: 模型原始输出字符串
    Returns:
        (最终答案文本或None, 原始字符串)
    """
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if not matches:
        print("[Error] No valid <answer>...</answer> tags found")
        return None, solution_str

    # 取最后一个 <answer>...</answer>
    final_answer = matches[-1].group(1).strip()
    return final_answer, solution_str


def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """解析数据集给定的标准答案文本为 {name: role} 字典。"""
    status_dict = {}
    print("\n[Ground Truth Parsing]")

    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Warning] Unparseable line: '{line}'")

    return status_dict


def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """从 <answer> 内文本解析出 {name: role}。要求覆盖全部 expected_names。"""
    status_dict = {}
    print("\n[Model Answer Parsing]")
    print(f"  Expected characters: {expected_names}")

    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b.*?\b(knight|knave)\b',
            re.IGNORECASE
        )
        match = pattern.search(answer_text)
        if match:
            role = match.group(1).lower()
            status_dict[name] = role
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Error] Missing identification for {name}")
            return None

    return status_dict


def validate_response_structure(processed_str: str) -> bool:
    """仅校验是否存在且仅存在一对 <answer>...</answer>，并检查顺序与配对。"""
    print("\n[Structure Validation]")
    validation_passed = True

    count_answer_start = processed_str.count('<answer>')
    count_answer_end = processed_str.count('</answer>')
    first_start = processed_str.find('<answer>')
    first_end = processed_str.find('</answer>')

    print(f"  <answer>: count={count_answer_start}, position={first_start}")
    print(f"  </answer>: count={count_answer_end}, position={first_end}")

    # 要求：恰好一对 <answer> 和 </answer>
    if count_answer_start != 1 or count_answer_end != 1:
        print("  [Error] There must be exactly one pair of <answer> and </answer>.")
        validation_passed = False

    # 顺序检查
    if first_start == -1 or first_end == -1 or first_start > first_end:
        print("  [Error] Incorrect order or missing tags for <answer>...</answer>.")
        validation_passed = False

    if validation_passed:
        print("  Answer tag validation passed")

    return validation_passed


def compute_score(data_source: Optional[Any],
                solution_str: str,
                ground_truth: Dict[str, str],
                extra_info: Optional[Dict[str, Any]] = None,
                format_reward: int = 1,
                answer_reward: float = 1.0):
    """根据仅有的 <answer>...</answer> 格式与答案正确性打分。

    评分规则：
    - 格式分：仅当存在且仅存在一对 <answer>...</answer> 且顺序正确时，得 format_reward，否则扣同等分值。
    - 内容分：在格式正确且能解析出全部人物身份时：
        * 若与标准答案完全一致：+2
        * 若能解析但与标准答案不一致：-1.5
        * 若无法解析：-2
      （保持你原先的数值）
    """
    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))

    # 解析标准答案
    solution_text = ground_truth.get('solution_text_format', '')
    gt_status = parse_solution_text_format(solution_text)
    expected_names = list(gt_status.keys())
    print(f"[Ground Truth] Final identities: {gt_status}")

    # 提取模型答案
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")

    # 仅验证 <answer> 标签结构
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    # 内容评分
    answer_score = 0
    if format_correct and answer_text:
        pred_status = parse_model_answer(answer_text, expected_names)
        if pred_status:
            print(f"\n[Content Validation]")
            print(f"  Expected: {gt_status}")
            print(f"  Predicted: {pred_status}")

            if pred_status == gt_status:
                answer_score = 2
                print("  Content validation: FULL MATCH")
            else:
                answer_score = -1.5
                print("  Content validation: MISMATCH")
        else:
            answer_score = -2
            print("  Content validation: FAIL to parse answer")
    else:
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")

    return total_score
