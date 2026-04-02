from __future__ import annotations

import json
from typing import Any

import numpy as np

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.agent_loop import register
from verl.tools.schemas import OpenAIFunctionToolSchema

from .tool import LocalFunctionTool


def _ensure_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    if isinstance(value, np.ndarray):
        return _ensure_dict(value.tolist())
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


def _normalize_single_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(schema)
    function = _ensure_dict(normalized.get("function"))
    parameters = _ensure_dict(function.get("parameters"))

    if parameters:
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            parameters["properties"] = {}
        parameters.setdefault("type", "object")
        parameters.setdefault("required", [])
        function["parameters"] = parameters

    normalized["function"] = function
    normalized.setdefault("type", "function")
    return normalized


@register("syntool_agent")
class SyntoolToolAgentLoop(ToolAgentLoop):
    """ToolAgentLoop variant that builds tools dynamically from each sample."""

    def _normalize_tool_schemas(self, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        extra_info = _ensure_dict(kwargs.get("extra_info"))

        for candidate in (
            kwargs.get("tool_schemas"),
            extra_info.get("tool_schemas"),
            extra_info.get("func_schemas"),
        ):
            schemas = []
            for schema in _ensure_list(candidate):
                if isinstance(schema, str):
                    try:
                        schema = json.loads(schema)
                    except json.JSONDecodeError:
                        continue
                if isinstance(schema, dict):
                    schemas.append(_normalize_single_tool_schema(schema))
            if schemas:
                return schemas
        return []

    def _build_tools_kwargs(self, kwargs: dict[str, Any], tool_schemas: list[dict[str, Any]]) -> dict[str, Any]:
        tools_kwargs = _ensure_dict(kwargs.get("tools_kwargs"))
        if tools_kwargs:
            return tools_kwargs

        extra_info = _ensure_dict(kwargs.get("extra_info"))
        tools_kwargs = _ensure_dict(extra_info.get("tools_kwargs"))
        if tools_kwargs:
            return tools_kwargs

        env = str(extra_info.get("env", "") or "")
        built_tools_kwargs = {}
        for schema in tool_schemas:
            tool_name = schema.get("function", {}).get("name")
            if tool_name:
                built_tools_kwargs[tool_name] = {"create_kwargs": {"env": env}}
        return built_tools_kwargs

    def _initialize_dynamic_tools(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        tool_schemas = self._normalize_tool_schemas(kwargs)
        tool_objects = []
        for schema in tool_schemas:
            schema_obj = OpenAIFunctionToolSchema.model_validate(schema)
            tool_objects.append(LocalFunctionTool(config={}, tool_schema=schema_obj))

        self.tools = {tool.name: tool for tool in tool_objects}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_objects]

        run_kwargs = dict(kwargs)
        extra_info = _ensure_dict(run_kwargs.get("extra_info"))
        extra_info["tool_schemas"] = self.tool_schemas
        if tool_schemas and "func_schemas" not in extra_info:
            extra_info["func_schemas"] = json.dumps(self.tool_schemas, ensure_ascii=False)
        run_kwargs["extra_info"] = extra_info
        run_kwargs["tool_schemas"] = self.tool_schemas
        run_kwargs["tools_kwargs"] = self._build_tools_kwargs(run_kwargs, self.tool_schemas)
        return run_kwargs

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        run_kwargs = self._initialize_dynamic_tools(kwargs)
        return await super().run(sampling_params=sampling_params, **run_kwargs)
