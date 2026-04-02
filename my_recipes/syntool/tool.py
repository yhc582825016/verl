from __future__ import annotations

import traceback
from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op


class LocalFunctionTool(BaseTool):
    """Execute mocked functions defined in per-sample env code."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config=config, tool_schema=tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str | None = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = f"local-func-{id(self)}-{len(self._instance_dict)}"
        create_kwargs = kwargs.get("create_kwargs", {}) or {}
        self._instance_dict[instance_id] = {"env": str(create_kwargs.get("env", "") or "")}
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        state = self._instance_dict.get(instance_id, {})
        env_code = str(state.get("env", "") or "")

        fn_name = (
            kwargs.get("function_name")
            or parameters.get("function_name")
            or parameters.get("function")
            or parameters.get("name")
            or self.name
        )
        fn_args = parameters.get("arguments") or parameters.get("args") or parameters
        if not isinstance(fn_args, dict):
            fn_args = {"value": fn_args}

        local_env: dict[str, Any] = {}
        try:
            exec(env_code, local_env, local_env)
        except Exception as exc:
            message = f"Error when loading env: {exc}\n{traceback.format_exc()}"
            return ToolResponse(text=message), 0.0, {}

        target_fn = local_env.get(fn_name)
        if not callable(target_fn):
            return ToolResponse(text=f"Error: function '{fn_name}' not found"), 0.0, {}

        try:
            result = target_fn(**fn_args)
            return ToolResponse(text=str(result)), None, {}
        except Exception as exc:
            message = f"Error when executing '{fn_name}': {exc}\n{traceback.format_exc()}"
            return ToolResponse(text=message), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
