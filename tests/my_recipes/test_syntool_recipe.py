from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from my_recipes.syntool.agent_loop import SyntoolToolAgentLoop
from my_recipes.syntool.dataset import SyntoolRLHFDataset
from my_recipes.syntool.reward import compute_syntool_score
from my_recipes.syntool.tool import LocalFunctionTool
from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.workers.rollout.replica import TokenOutput


TEST_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "first number"},
                "b": {"type": "integer", "description": "second number"},
            },
            "required": ["a", "b"],
        },
    },
}


class _SimpleTokenizer:
    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "</s>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(chr(int(token)) for token in ids if int(token) > 0)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ):
        del kwargs
        parts = []
        if tools:
            parts.append("<tools>\n")
            parts.extend(json.dumps(tool, ensure_ascii=False) + "\n" for tool in tools)
            parts.append("</tools>\n")

        for message in messages:
            role = message["role"]
            content = message["content"]
            if isinstance(content, list):
                content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
            if role == "tool":
                parts.append(f"<tool_response>{content}</tool_response>")
            else:
                parts.append(f"<{role}>{content}</{role}>")

        if add_generation_prompt:
            parts.append("<assistant>")

        text = "".join(parts)
        return self.encode(text) if tokenize else text

    def pad(
        self,
        encoded_inputs: dict[str, list[int]],
        *,
        padding: str,
        max_length: int,
        return_tensors: str,
        return_attention_mask: bool,
    ) -> dict[str, torch.Tensor]:
        del padding, return_tensors
        input_ids = encoded_inputs["input_ids"]
        if len(input_ids) > max_length:
            if self.padding_side == "left":
                input_ids = input_ids[-max_length:]
            else:
                input_ids = input_ids[:max_length]

        pad_len = max_length - len(input_ids)
        if self.padding_side == "left":
            padded_ids = [self.pad_token_id] * pad_len + input_ids
            attention_mask = [0] * pad_len + [1] * len(input_ids)
        else:
            padded_ids = input_ids + [self.pad_token_id] * pad_len
            attention_mask = [1] * len(input_ids) + [0] * pad_len

        output = {"input_ids": torch.tensor([padded_ids], dtype=torch.long)}
        if return_attention_mask:
            output["attention_mask"] = torch.tensor([attention_mask], dtype=torch.long)
        return output


class _StaticDecodeTokenizer:
    def __init__(self, text: str):
        self.text = text

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return self.text


class _FakeServerManager:
    def __init__(self, tokenizer: _SimpleTokenizer):
        self.tokenizer = tokenizer
        self.call_count = 0

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        del request_id, prompt_ids, sampling_params, image_data, video_data
        self.call_count += 1
        if self.call_count == 1:
            text = (
                "<tool_call>\n"
                "<function=add>\n"
                "<parameter=a>\n40\n</parameter>\n"
                "<parameter=b>\n2\n</parameter>\n"
                "</function>\n"
                "</tool_call>"
            )
        else:
            text = "The final answer is //box{42}"

        token_ids = self.tokenizer.encode(text)
        return TokenOutput(token_ids=token_ids, log_probs=[0.0] * len(token_ids))


def _make_config(**data_overrides):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 256,
                    "response_length": 256,
                    "multi_turn": {
                        "tool_config_path": None,
                        "interaction_config_path": None,
                        "max_user_turns": 8,
                        "max_assistant_turns": 8,
                        "max_parallel_calls": 1,
                        "max_tool_response_length": 512,
                        "tool_response_truncate_side": "middle",
                        "format": "qwen3_coder",
                    },
                },
                "model": {},
            },
            "data": {
                "prompt_key": "prompt",
                "image_key": "images",
                "video_key": "videos",
                "max_prompt_length": 512,
                "return_raw_chat": True,
                "filter_overlong_prompts": False,
                "truncation": "error",
                "tool_config_path": None,
                "apply_chat_template_kwargs": {},
                **data_overrides,
            },
        }
    )


@pytest.fixture
def syntool_row() -> dict[str, Any]:
    env = "def add(a, b):\n    return a + b\n"
    return {
        "data_source": "syntool",
        "question": "What is 40 + 2?",
        "ability": "re_call",
        "reward_model": {"ground_truth": ["42"], "style": "rule"},
        "extra_info": json.dumps(
            {
                "env": env,
                "func_schemas": json.dumps([TEST_TOOL_SCHEMA]),
                "index": 7,
            }
        ),
    }


def test_syntool_dataset_normalizes_row(tmp_path: Path, syntool_row):
    file_path = tmp_path / "syntool.parquet"
    pd.DataFrame([syntool_row]).to_parquet(file_path)

    dataset = SyntoolRLHFDataset(
        data_files=[str(file_path)],
        tokenizer=_SimpleTokenizer(),
        config=_make_config().data,
    )

    row = dataset[0]
    assert row["agent_name"] == "syntool_agent"
    assert row["reward_model"]["ground_truth"] == ["42"]
    assert row["tool_schemas"][0]["function"]["name"] == "add"
    assert row["tools_kwargs"]["add"]["create_kwargs"]["env"].startswith("def add")
    assert row["extra_info"]["index"] == 7
    assert row["raw_prompt"][0]["role"] == "system"
    assert "//box{final_answer}" in row["raw_prompt"][0]["content"]


@pytest.mark.asyncio
async def test_local_function_tool_executes_mocked_env():
    schema = OpenAIFunctionToolSchema.model_validate(TEST_TOOL_SCHEMA)
    tool = LocalFunctionTool(config={}, tool_schema=schema)

    instance_id, _ = await tool.create(create_kwargs={"env": "def add(a, b):\n    return a + b\n"})
    response, reward, _ = await tool.execute(instance_id, {"a": 1, "b": 2})
    assert response.text == "3"
    assert reward is None

    missing_instance_id, _ = await tool.create(create_kwargs={"env": "def other(a, b):\n    return a + b\n"})
    missing_response, missing_reward, _ = await tool.execute(missing_instance_id, {"a": 1, "b": 2})
    assert "not found" in (missing_response.text or "")
    assert missing_reward == 0.0


def test_compute_syntool_score_supports_box_variants():
    result = compute_syntool_score(
        data_source="syntool",
        solution_str="thinking <tool_call>ignored</tool_call><tool_response>ignored</tool_response> //box{42}",
        ground_truth=["42"],
        extra_info={},
    )
    assert result["score"] == 1.0
    assert result["pred"] == "42"

    boxed = compute_syntool_score(
        data_source="syntool",
        solution_str="Final: \\boxed{42}",
        ground_truth=["42"],
        extra_info={},
    )
    assert boxed["score"] == 1.0

    special = compute_syntool_score(
        data_source="syntool",
        solution_str="Final: <|box_start|>42<|box_end|>",
        ground_truth=["42"],
        extra_info={},
    )
    assert special["score"] == 1.0

    fallback = compute_syntool_score(
        data_source="syntool",
        solution_str="<tool_call>ignored</tool_call><tool_response>ignored</tool_response> 42 ",
        ground_truth=["42"],
        extra_info={},
    )
    assert fallback["score"] == 1.0


@pytest.mark.asyncio
async def test_qwen3_coder_parser_extracts_parameters():
    parser = ToolParser.get_tool_parser(
        "qwen3_coder",
        _StaticDecodeTokenizer(
            "<tool_call>\n"
            "<function=add>\n"
            "<parameter=a>\n40\n</parameter>\n"
            "<parameter=b>\n2\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
    )
    schemas = [OpenAIFunctionToolSchema.model_validate(TEST_TOOL_SCHEMA)]
    content, calls = await parser.extract_tool_calls([1, 2, 3], schemas)
    assert content == ""
    assert len(calls) == 1
    assert calls[0].name == "add"
    assert json.loads(calls[0].arguments) == {"a": 40, "b": 2}


@pytest.mark.asyncio
async def test_syntool_agent_loop_runs_multiturn_function_call():
    tokenizer = _SimpleTokenizer()
    server_manager = _FakeServerManager(tokenizer)
    config = _make_config()

    agent_loop = SyntoolToolAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=SyntoolRLHFDataset,
        data_config=DictConfigWrap(config.data),
    )

    env = "def add(a, b):\n    return a + b\n"
    output = await agent_loop.run(
        sampling_params={},
        raw_prompt=[
            {"role": "system", "content": "Use tools if needed and answer with //box{...}."},
            {"role": "user", "content": "What is 40 + 2?"},
        ],
        tool_schemas=[TEST_TOOL_SCHEMA],
        tools_kwargs={"add": {"create_kwargs": {"env": env}}},
        extra_info={"env": env, "tool_schemas": [TEST_TOOL_SCHEMA]},
    )

    response_text = tokenizer.decode(output.response_ids)
    assert server_manager.call_count == 2
    assert "//box{42}" in response_text
    assert "<tool_response>42</tool_response>" in response_text
    assert 0 in output.response_mask
    assert output.num_turns == 4
    assert output.extra_fields["turn_scores"] == []
    assert output.extra_fields["tool_rewards"] == []


def test_qwen35_local_chat_template_has_expected_tool_format():
    model_path = Path("/opt/users/models/Qwen3.5-9B")
    if not model_path.exists():
        pytest.skip("Local Qwen3.5 tokenizer is not available")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Answer with //box{...}."},
            {"role": "user", "content": "What is 40 + 2?"},
        ],
        tools=[TEST_TOOL_SCHEMA],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert "<tools>" in text
    assert "<tool_call>" in text
    assert "<function=example_function_name>" in text
    assert "<tool_response>" not in text
