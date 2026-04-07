"""SpinalCord Bench environment — text tasks with deterministic graders."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from spinalcord_bench.models import (
    Difficulty,
    SpinalBenchAction,
    SpinalBenchObservation,
    SpinalBenchState,
)
from spinalcord_bench.server.graders import GRADERS, TASKS, pick_task_id


def _norm_reply(s: str) -> str:
    return " ".join(s.strip().lower().split())


class SpinalBenchEnv(Environment[SpinalBenchAction, SpinalBenchObservation, SpinalBenchState]):
    """LLM-facing benchmark: each step submits an assistant string; env returns graded observation."""

    MAX_STEPS = 6

    def __init__(self) -> None:
        super().__init__()
        self._state = SpinalBenchState()
        self._task_id: str = TASKS["extract_total"].task_id
        self._instruction: str = ""
        self._context: str = ""
        self._difficulty: Difficulty = "easy"

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SpinalBenchObservation:
        self._reset_rubric()
        self._task_id = pick_task_id(seed, task_id)
        spec = TASKS[self._task_id]
        self._instruction = spec.instruction
        self._context = spec.reference_context
        self._difficulty = spec.difficulty  # type: ignore[assignment]

        eid = episode_id or str(uuid.uuid4())
        self._state = SpinalBenchState(
            episode_id=eid,
            step_count=0,
            task_id=self._task_id,
            difficulty=self._difficulty,
            last_reply_norm="",
            terminal=False,
            cumulative_grader=0.0,
        )

        return SpinalBenchObservation(
            instruction=self._instruction,
            task_id=self._task_id,
            difficulty=self._difficulty,
            reference_context=self._context,
            grader_score=0.0,
            grader_detail="episode_started",
            reward=0.0,
            done=False,
            max_steps=self.MAX_STEPS,
            step_limit_reached=False,
            metadata={"seed": seed, "episode_id": eid},
        )

    def step(
        self,
        action: SpinalBenchAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SpinalBenchObservation:
        text = action.content.strip()
        grader = GRADERS[self._task_id]
        score, detail = grader(text)

        # Shaping: incremental signal + penalties (trajectory-level)
        loop_penalty = 0.0
        empty_penalty = 0.0
        if not text:
            empty_penalty = 0.12
        prev = self._state.last_reply_norm
        cur = _norm_reply(text)
        if prev and cur == prev:
            loop_penalty = 0.18
        self._state.last_reply_norm = cur

        shaped = 0.35 * score - empty_penalty - loop_penalty
        self._state.step_count += 1
        self._state.cumulative_grader += score

        at_limit = self._state.step_count >= self.MAX_STEPS
        solved = score >= 0.99
        done = solved or at_limit
        self._state.terminal = done

        terminal_bonus = 0.0
        if done and solved:
            terminal_bonus = 0.45
        elif done and not solved:
            terminal_bonus = -0.05

        reward = float(shaped + terminal_bonus)

        return SpinalBenchObservation(
            instruction=self._instruction,
            task_id=self._task_id,
            difficulty=self._difficulty,
            reference_context=self._context,
            grader_score=score,
            grader_detail=detail,
            reward=reward,
            done=done,
            max_steps=self.MAX_STEPS,
            step_limit_reached=at_limit and not solved,
            metadata={
                "empty_penalty": empty_penalty,
                "loop_penalty": loop_penalty,
                "terminal_bonus": terminal_bonus,
            },
        )

    @property
    def state(self) -> SpinalBenchState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="SpinalCordBench",
            description=(
                "Synthetic text tasks (invoice, calendar, logs) with deterministic "
                "graders for evaluating OpenAI-compatible endpoints (e.g. SpinalCord "
                "via llama-server)."
            ),
            version="1.0.0",
            author="SpinalCord project",
            documentation_url="https://github.com/Shivanshdarji/Spinal-cord",
        )
