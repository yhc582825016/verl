from __future__ import annotations

import json
import re
from typing import Any


_TOOL_BLOCK_PATTERNS = (
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"<tool_response>.*?</tool_response>", re.DOTALL),
)


def _normalize_text(text: Any) -> str:
    return str(text or "").strip().lower()


def _clean_solution_text(text: str) -> str:
    cleaned = text or ""
    for pattern in _TOOL_BLOCK_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_last_braced(text: str, marker: str) -> str | None:
    start = text.rfind(marker)
    if start < 0:
        return None

    index = start + len(marker) - 1
    open_braces = 0
    end = None
    while index < len(text):
        char = text[index]
        if char == "{":
            open_braces += 1
        elif char == "}":
            open_braces -= 1
            if open_braces == 0:
                end = index
                break
        index += 1

    if end is None:
        return None
    return text[start + len(marker) : end]


def _extract_last_special_box(text: str) -> str | None:
    matches = re.findall(r"<\|box_start\|>(.*?)<\|box_end\|>", text, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


def _extract_prediction(solution_str: str) -> str:
    final_answer_text = _clean_solution_text(solution_str)

    for marker in ("//box{", "\\boxed{"):
        extracted = _extract_last_braced(final_answer_text, marker)
        if extracted is not None:
            return extracted.strip()

    special_box = _extract_last_special_box(final_answer_text)
    if special_box is not None:
        return special_box

    return "提取失败"


def _normalize_targets(ground_truth: Any) -> list[str]:
    if ground_truth is None:
        return [""]

    if isinstance(ground_truth, str):
        return [_normalize_text(ground_truth)]

    try:
        values = list(ground_truth)
    except TypeError:
        return [_normalize_text(ground_truth)]

    if not values:
        return [""]
    return [_normalize_text(value) for value in values]


def _to_json_safe(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _to_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    return value


def compute_syntool_score(data_source: str, solution_str: str, ground_truth, extra_info=None):
    del data_source

    extra_info = extra_info or {}
    tool_extra_fields = extra_info.get("tool_extra_fields", {}) or {}

    final_answer_text = _clean_solution_text(solution_str)
    prediction = _extract_prediction(solution_str)
    normalized_prediction = _normalize_text(prediction)
    targets = _normalize_targets(ground_truth)
    hit = any(target == normalized_prediction for target in targets)
    score = 1.0 if hit else 0.0
    primary_ground_truth = ground_truth[0] if isinstance(ground_truth, list) and ground_truth else ground_truth

    log_payload = {
        "prompt_str": extra_info.get("prompt_str"),
        "solution_str": solution_str,
        # "final_answer_text": final_answer_text[:200],
        "prediction": prediction[:200],
        "ground_truth": ground_truth,
        "score": score,
        "num_turns": extra_info.get("num_turns"),
        # "termination_reason": tool_extra_fields.get("tool_debug_termination_reason"),
        # "parsed_tool_call_count": tool_extra_fields.get("tool_debug_parsed_tool_call_count"),
        # "has_tool_call_markup": tool_extra_fields.get("tool_debug_has_tool_call_markup"),
        # "decoded_response_length": tool_extra_fields.get("tool_debug_decoded_response_length"),
        # "response_length_limit": tool_extra_fields.get("tool_debug_response_length_limit"),
        # "assistant_turns": tool_extra_fields.get("tool_debug_assistant_turns"),
        # "user_turns": tool_extra_fields.get("tool_debug_user_turns"),
        # "max_assistant_turns_limit": tool_extra_fields.get("tool_debug_max_assistant_turns_limit"),
        # "max_user_turns_limit": tool_extra_fields.get("tool_debug_max_user_turns_limit"),
    }
    print(json.dumps(_to_json_safe(log_payload), ensure_ascii=False))

    return {
        "score": score,
        "ground_truth": primary_ground_truth if primary_ground_truth is not None else "",
        "prediction": prediction,
        "target": targets[0] if targets else "",
        "pred": prediction,
        "acc": hit,
    }
