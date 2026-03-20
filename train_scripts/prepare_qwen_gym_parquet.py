#!/usr/bin/env python3
"""Convert qwen_gym jsonl data to verl parquet format.

Input rows are expected to be similar to:
- ms-swift/qwen_gym/dapo_math_17k_swift_nemo_gym.jsonl
- ms-swift/qwen_gym/aime_2024_swift_nemo_gym.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _build_prompt(row: dict[str, Any]) -> list[dict[str, Any]]:
    responses_create_params = row.get("responses_create_params", {}) or {}
    inputs = responses_create_params.get("input", None)
    if isinstance(inputs, list) and inputs:
        return inputs

    question = row.get("question") or row.get("problem") or ""
    return [{"role": "user", "content": str(question)}]


def _convert_row(
    row: dict[str, Any],
    index: int,
    verify_url: str,
    default_data_source: str,
) -> dict[str, Any]:
    env_config = dict(row.get("env_config") or {})
    env_config.setdefault("name", "nemo_gym_env")
    env_config["verify_url"] = verify_url

    reward_model = dict(row.get("reward_model") or {})
    if not reward_model.get("ground_truth"):
        reward_model["ground_truth"] = row.get("solution", row.get("expected_answer", ""))

    data_source = row.get("data_source", default_data_source)

    extra_info = dict(row.get("extra_info") or {})
    extra_info["index"] = int(extra_info.get("index", index))
    extra_info["interaction_kwargs"] = {
        "name": "gym",
        "env_config": env_config,
    }

    converted = {
        "data_source": data_source,
        "prompt": _build_prompt(row),
        "ability": row.get("ability", "math"),
        "reward_model": reward_model,
        "extra_info": extra_info,
        # Keep these fields for gym env payload assembly.
        "responses_create_params": row.get("responses_create_params", {}),
        "question": row.get("question", row.get("problem", "")),
        "problem": row.get("problem", row.get("question", "")),
        "solution": row.get("solution", row.get("expected_answer", "")),
        "expected_answer": row.get("expected_answer", row.get("solution", "")),
        "env_config": env_config,
        "ctx_config": row.get("ctx_config", {}),
    }
    return converted


def convert_jsonl_to_parquet(
    input_path: Path,
    output_path: Path,
    verify_url: str,
    default_data_source: str,
) -> int:
    rows = _load_jsonl(input_path)
    converted = [
        _convert_row(row=row, index=i, verify_url=verify_url, default_data_source=default_data_source)
        for i, row in enumerate(rows)
    ]
    ds = Dataset.from_list(converted)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(output_path))
    return len(ds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert qwen_gym JSONL to verl parquet.")
    parser.add_argument(
        "--train_jsonl",
        required=True,
        help="Input train JSONL file.",
    )
    parser.add_argument(
        "--val_jsonl",
        required=True,
        help="Input val JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for train.parquet and val.parquet.",
    )
    parser.add_argument(
        "--verify_url",
        default="http://127.0.0.1:18001/verify",
        help="Gym verifier URL.",
    )
    parser.add_argument(
        "--default_data_source",
        default="math_gym",
        help="Fallback data_source for rows missing this field.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"

    train_n = convert_jsonl_to_parquet(
        input_path=Path(args.train_jsonl).expanduser().resolve(),
        output_path=train_path,
        verify_url=args.verify_url,
        default_data_source=args.default_data_source,
    )
    val_n = convert_jsonl_to_parquet(
        input_path=Path(args.val_jsonl).expanduser().resolve(),
        output_path=val_path,
        verify_url=args.verify_url,
        default_data_source=args.default_data_source,
    )
    print(f"train rows={train_n} -> {train_path}")
    print(f"val rows={val_n} -> {val_path}")


if __name__ == "__main__":
    main()

