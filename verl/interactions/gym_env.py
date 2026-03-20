"""Gym-style environment abstractions for multi-turn interactions.

This module mirrors the lightweight Env/ContextManager interfaces used in
other RL stacks so users can register custom gym-like environments directly
in verl.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import aiohttp


Messages = list[dict[str, Any]]


class ContextManager(ABC):
    """Base interface for conversation context management."""

    def __init__(self, ctx_config: dict[str, Any] | None = None):
        self.ctx_config = ctx_config or {}

    @abstractmethod
    def manage_context(self, history: Messages, trajectory_id: str) -> Messages:
        """Return managed history for the next rollout turn."""


class DummyContextManager(ContextManager):
    """No-op context manager."""

    def manage_context(self, history: Messages, trajectory_id: str) -> Messages:
        return history


class Env(ABC):
    """Base gym-style environment interface."""

    def __init__(self, env_config: dict[str, Any] | None = None):
        self.env_config = env_config or {}

    @abstractmethod
    async def reset(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        """Reset environment and return (observation, info, system_message)."""

    @abstractmethod
    async def step(self, action: Messages) -> tuple[str, float, bool, dict[str, Any]]:
        """Advance one step and return (next_observation, reward, done, info)."""

    @abstractmethod
    async def close(self) -> None:
        """Release environment resources."""


class NemoGymVerifyEnv(Env):
    """Single-turn verifier-backed gym environment.

    This env is designed for NeMo Gym style `/verify` endpoints. It scores the
    latest assistant response against server-side verification logic and can be
    configured to terminate immediately after one scoring call.
    """

    def __init__(self, env_config: dict[str, Any] | None = None):
        super().__init__(env_config)
        self.verify_url = self.env_config.get("verify_url")
        if not self.verify_url:
            raise ValueError("`verify_url` is required for `nemo_gym_env`.")

        self.prompt_key = self.env_config.get("prompt_key", "question")
        self.reward_key = self.env_config.get("reward_key", "reward")
        self.done_on_verify = bool(self.env_config.get("done_on_verify", True))
        self.next_observation = self.env_config.get(
            "next_observation",
            "Please continue and improve your answer.",
        )
        self.system_prompt = self.env_config.get("system_prompt", "")
        self.request_timeout = float(self.env_config.get("request_timeout", 30.0))
        self._sample: dict[str, Any] = {}
        self._session: aiohttp.ClientSession | None = None

    async def reset(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        sample = payload.get("sample", {}) or {}
        self._sample = sample

        question = (
            sample.get(self.prompt_key)
            or sample.get("question")
            or sample.get("problem")
            or payload.get("question")
            or ""
        )
        info = {
            "env": "nemo_gym_env",
            "verify_url": self.verify_url,
        }
        return str(question), info, self.system_prompt

    async def step(self, action: Messages) -> tuple[str, float, bool, dict[str, Any]]:
        payload = self._build_verify_payload(action)
        response_json = await self._post_json(payload)

        reward = response_json.get(self.reward_key, 0.0)
        try:
            reward = float(reward)
        except (TypeError, ValueError):
            reward = 0.0

        done = bool(response_json.get("done", self.done_on_verify))
        next_observation = str(response_json.get("observation", self.next_observation))
        info = {
            "verify_response": response_json,
            "stop_reason": "verified" if done else "continue",
        }
        return next_observation, reward, done, info

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _build_verify_payload(self, action: Messages) -> dict[str, Any]:
        response_text = ""
        if action:
            for msg in reversed(action):
                if msg.get("role") == "assistant":
                    response_text = str(msg.get("content", ""))
                    break

        request_messages = self._sample.get("responses_create_params", {}).get("input")
        if not request_messages:
            request_messages = self._sample.get("raw_prompt")
        if not request_messages:
            request_messages = [m for m in action if m.get("role") in ("system", "user")]
        request_messages = list(request_messages)
        request_messages.append({"role": "assistant", "content": response_text})

        body = {
            "responses_create_params": {
                "input": request_messages,
            },
            "reward_model": self._sample.get("reward_model", {}),
            "data_source": self._sample.get("data_source", ""),
            "extra_info": self._sample.get("extra_info", {}),
            "question": self._sample.get("question", self._sample.get("problem", "")),
            "solution": self._sample.get("solution", ""),
            "expected_answer": self._sample.get("expected_answer", ""),
            "env_config": self.env_config,
        }
        return body

    async def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)

        async with self._session.post(self.verify_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


context_managers: dict[str, type[ContextManager]] = {
    "dummyContextManager": DummyContextManager,
}

envs: dict[str, type[Env]] = {
    "nemo_gym_env": NemoGymVerifyEnv,
}

