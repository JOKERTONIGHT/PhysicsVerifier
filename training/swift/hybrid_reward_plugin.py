#!/usr/bin/env python3
"""ms-swift GRPO plugin: outcome-gated hybrid of DeepSeek process score + local acc.

Gold labels are sent to the reward server only. The server must never forward them
to the DeepSeek judge prompt.
"""
from __future__ import annotations

import asyncio
import os
from typing import List

from training.swift.llm_step_reward_plugin import (
    LLMStepVerifierReward,
    REWARD_URL,
    TIMEOUT,
    build_reward_payload,
    group_indices_by_prompt,
    orms,
    slice_payload,
)


class HybridLLMOutcomeReward(LLMStepVerifierReward):
    """Same HTTP transport as llm_step_verifier, but labels travel with the payload."""

    async def __call__(
        self,
        completions,
        messages=None,
        solution=None,
        question=None,
        **kwargs,
    ) -> List[float]:
        payload = build_reward_payload(
            completions,
            messages=messages,
            question=question if question is not None else kwargs.get("question"),
            include_labels=True,
            solution=solution if solution is not None else kwargs.get("solution"),
        )
        if any(not str(label or "").strip() for label in payload["labels"]):
            raise RuntimeError("hybrid_llm_outcome requires labels; refusing process-only fallback")
        import aiohttp
        n = len(payload["query"])
        timeout = aiohttp.ClientTimeout(total=float(os.environ.get("PHYSICS_REWARD_TIMEOUT", str(TIMEOUT))))
        retries = int(os.environ.get("PHYSICS_REWARD_HTTP_RETRIES", "5"))
        url = os.environ.get("PHYSICS_REWARD_URL", REWARD_URL)
        groups = group_indices_by_prompt(payload["prompts"])
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if len(groups) <= 1:
                return await self._post_payload(session, url, payload, retries)
            scored = await asyncio.gather(
                *[self._post_payload(session, url, slice_payload(payload, idxs), retries) for idxs in groups]
            )
        out = [0.0] * n
        for idxs, rewards in zip(groups, scored):
            for i, reward in zip(idxs, rewards):
                out[i] = reward
        return out


orms["hybrid_llm_outcome"] = HybridLLMOutcomeReward
orms["outcome_only"] = HybridLLMOutcomeReward
