from __future__ import annotations

import copy
import json
import logging
import os
import traceback
from typing import Any

import numpy as np
import pandas as pd
import torch

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "Use the provided tools whenever they help you solve the task.\n"
    "After you finish using tools, output the final answer in the format //box{final_answer}."
)


def _normalize_text(text: Any) -> str:
    if isinstance(text, np.ndarray):
        text = text.tolist()
    return str(text if text is not None else "")


def _ensure_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, np.ndarray):
        return _ensure_dict(value.tolist())
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return _ensure_list(value.tolist())
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return _ensure_list(loaded)
    if isinstance(value, dict):
        return [value]
    return [value]


class SyntoolRLHFDataset(RLHFDataset):
    """Dataset adapter for syntool multi-turn mocked function-calling data."""

    @staticmethod
    def _load_json_rows(file_path: str) -> list[dict[str, Any]] | None:
        path = str(file_path)
        if path.endswith(".jsonl"):
            rows = []
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows

        if path.endswith(".json"):
            with open(path, encoding="utf-8") as handle:
                content = json.load(handle)
            if isinstance(content, list):
                return content
            if isinstance(content, dict):
                for key in ("data", "train", "examples", "items", "rows", "records"):
                    if isinstance(content.get(key), list):
                        return content[key]
                return [content]
        return None

    @staticmethod
    def _normalize_ground_truth(ground_truth: Any) -> list[str]:
        if ground_truth is None:
            return [""]
        if isinstance(ground_truth, str):
            return [ground_truth]
        if isinstance(ground_truth, np.ndarray):
            return [str(item) for item in ground_truth.tolist()]
        try:
            values = list(ground_truth)
        except TypeError:
            return [str(ground_truth)]
        if not values:
            return [""]
        return [str(item) for item in values]

    @staticmethod
    def _normalize_tool_schemas(extra_info: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = (
            extra_info.get("tool_schemas"),
            extra_info.get("func_schemas"),
        )
        for candidate in candidates:
            schemas = []
            for schema in _ensure_list(candidate):
                if isinstance(schema, str):
                    try:
                        schema = json.loads(schema)
                    except json.JSONDecodeError:
                        continue
                if isinstance(schema, dict):
                    schemas.append(schema)
            if schemas:
                return schemas
        return []

    @staticmethod
    def _extract_question(row: dict[str, Any]) -> str:
        question = row.get("question", row.get("prompt", ""))
        if isinstance(question, np.ndarray):
            question = question.tolist()
        if isinstance(question, list) and question:
            first = question[0]
            if isinstance(first, dict) and "content" in first:
                question = first["content"]
        return _normalize_text(question)

    def _build_tools_kwargs(self, tool_schemas: list[dict[str, Any]], env: str) -> dict[str, Any]:
        tools_kwargs = {}
        for schema in tool_schemas:
            tool_name = schema.get("function", {}).get("name")
            if tool_name:
                tools_kwargs[tool_name] = {"create_kwargs": {"env": env}}
        return tools_kwargs

    def _map_row(self, row: dict[str, Any], index: int) -> dict[str, Any]:
        data_source = _normalize_text(row.get("data_source", "syntool"))
        ability = _normalize_text(row.get("ability", "tool_use"))
        question = self._extract_question(row)

        reward_model = _ensure_dict(row.get("reward_model"))
        reward_model["ground_truth"] = self._normalize_ground_truth(reward_model.get("ground_truth"))
        reward_model.setdefault("style", "rule")

        extra_info = _ensure_dict(row.get("extra_info"))
        env = _normalize_text(extra_info.get("env", ""))
        tool_schemas = self._normalize_tool_schemas(extra_info)
        tools_kwargs = self._build_tools_kwargs(tool_schemas, env)

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        normalized_extra_info = dict(extra_info)
        normalized_extra_info["index"] = normalized_extra_info.get("index", index)
        normalized_extra_info["env"] = env
        normalized_extra_info["tool_schemas"] = tool_schemas
        normalized_extra_info["func_schemas"] = json.dumps(tool_schemas, ensure_ascii=False)
        normalized_extra_info["tools_kwargs"] = tools_kwargs
        normalized_extra_info["need_tools_kwargs"] = True

        return {
            "data_source": data_source,
            "prompt": prompt,
            "ability": ability,
            "reward_model": reward_model,
            "extra_info": normalized_extra_info,
            "tool_schemas": tool_schemas,
            "agent_name": "syntool_agent",
        }

    def _prompt_length(self, row: dict[str, Any]) -> int:
        try:
            apply_kwargs = dict(self.apply_chat_template_kwargs)
            apply_kwargs.pop("tokenize", None)
            apply_kwargs.pop("return_dict", None)
            apply_kwargs.pop("return_tensors", None)

            messages = self._build_messages(copy.deepcopy(row))
            tools = row.get("tool_schemas") or None

            if self.processor is not None:
                raw_prompt = self.processor.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **apply_kwargs,
                )
                tokenized = self.processor.tokenizer(
                    text=raw_prompt,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
                return len(tokenized)

            tokenized_prompt = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                **apply_kwargs,
            )
            return len(normalize_token_ids(tokenized_prompt))
        except Exception:
            logger.warning("Failed to measure syntool prompt length:\n%s", traceback.format_exc())
            return self.max_prompt_length + 1

    def _maybe_filter_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.filter_overlong_prompts:
            return rows

        filtered_rows = [row for row in rows if self._prompt_length(row) <= self.max_prompt_length]
        print(f"filter dataset len: {len(filtered_rows)}")
        return filtered_rows

    def _read_files_and_tokenize(self):
        processed_rows: list[dict[str, Any]] = []

        for data_file in self.data_files:
            data_file = str(data_file)
            raw_rows = self._load_json_rows(data_file)
            if raw_rows is None:
                if data_file.endswith(".parquet"):
                    raw_rows = pd.read_parquet(data_file).to_dict("records")
                else:
                    raise ValueError(f"Unsupported file format for syntool dataset: {data_file}")

            for row in raw_rows:
                processed_rows.append(self._map_row(row, len(processed_rows)))

        total = len(processed_rows)
        print(f"syntool dataset len: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
                processed_rows = [processed_rows[int(idx)] for idx in indices]
            else:
                processed_rows = processed_rows[: self.max_samples]
            print(f"selected {len(processed_rows)} random samples out of {total}")

        self.dataframe = self._maybe_filter_rows(processed_rows)

    def __getitem__(self, item):
        row_dict = copy.deepcopy(self.dataframe[item])
        row_dict["raw_prompt"] = self._build_messages(row_dict)
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        extra_info = row_dict.get("extra_info") or {}
        row_dict["extra_info"] = extra_info
        row_dict["index"] = extra_info.get("index", 0)
        row_dict["tools_kwargs"] = extra_info.get("tools_kwargs", {})
        row_dict["interaction_kwargs"] = extra_info.get("interaction_kwargs", {})
        return row_dict
