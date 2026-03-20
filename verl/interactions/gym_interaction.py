from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional
from uuid import uuid4

from .base import BaseInteraction
from .gym_env import ContextManager, DummyContextManager, Env, context_managers, envs


class GymInteraction(BaseInteraction):
    """Gym-style interaction wrapper for tool-agent multi-turn rollout."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self.default_env_name = config.get("default_env", None)
        self.default_context_manager = config.get("default_context_manager", "dummyContextManager")
        self.use_env_reset_messages = bool(config.get("use_env_reset_messages", False))

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        env_config: Optional[dict[str, Any]] = None,
        ctx_config: Optional[dict[str, Any]] = None,
        initial_messages: Optional[list[dict[str, Any]]] = None,
        sample: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        sample = sample or {}
        env_config = dict(sample.get("env_config", {})) | dict(env_config or {})
        ctx_config = dict(sample.get("ctx_config", {})) | dict(ctx_config or {})

        env = self._create_env(env_config)
        context_manager = self._create_context_manager(ctx_config)

        observation, info, system_message = await env.reset(
            {
                "sample": sample,
                "kwargs": kwargs,
            }
        )

        boot_messages: list[dict[str, Any]] = list(initial_messages or [])
        if self.use_env_reset_messages or not boot_messages:
            boot_messages = []
            if system_message:
                boot_messages.append({"role": "system", "content": system_message})
            if observation:
                boot_messages.append({"role": "user", "content": observation})

        self._instance_dict[instance_id] = {
            "env": env,
            "context_manager": context_manager,
            "messages": boot_messages,
            "step_rewards": [],
            "trajectory_info": [info],
            "total_reward": 0.0,
        }
        return instance_id

    async def get_initial_messages(self, instance_id: str) -> list[dict[str, Any]]:
        state = self._instance_dict[instance_id]
        return list(state["messages"])

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        state = self._instance_dict[instance_id]
        env: Env = state["env"]
        context_manager: ContextManager = state["context_manager"]

        managed_messages = context_manager.manage_context(list(messages), instance_id)
        next_observation, reward, done, info = await env.step(deepcopy(managed_messages))

        state["messages"] = managed_messages
        state["total_reward"] += reward
        state["step_rewards"].append(reward)
        state["trajectory_info"].append(info)

        metrics = {
            "managed_messages": managed_messages,
            "trajectory_info": state["trajectory_info"],
            "step_rewards": state["step_rewards"],
            "total_reward": state["total_reward"],
        }
        if info:
            metrics.update(info)

        if done:
            await self.finalize_interaction(instance_id)
        return done, next_observation, reward, metrics

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        state = self._instance_dict.pop(instance_id, None)
        if not state:
            return
        env: Env = state["env"]
        await env.close()

    def _create_env(self, env_config: dict[str, Any]) -> Env:
        env_name = env_config.get("name", self.default_env_name)
        if not env_name:
            raise ValueError("GymInteraction requires `env_config.name` or `default_env`.")
        if env_name not in envs:
            raise ValueError(f"Environment '{env_name}' not found. Available: {list(envs.keys())}")
        return envs[env_name](env_config)

    def _create_context_manager(self, ctx_config: dict[str, Any]) -> ContextManager:
        ctx_name = ctx_config.get("name", self.default_context_manager)
        if not ctx_name:
            return DummyContextManager({})
        if ctx_name not in context_managers:
            raise ValueError(f"Context manager '{ctx_name}' not found. Available: {list(context_managers.keys())}")
        return context_managers[ctx_name](ctx_config)

