"""Pydantic wire types for SpinalCord Bench."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State

Difficulty = Literal["easy", "medium", "hard"]


class SpinalBenchAction(Action):
    """Assistant turn: natural-language answer to the current instruction."""

    content: str = Field(
        ...,
        min_length=1,
        description="Model reply (what you would send as assistant message).",
    )


class SpinalBenchObservation(Observation):
    """Observation with task text, grading feedback, and shaping metadata."""

    instruction: str = Field(..., description="What the agent must do this episode.")
    task_id: str = Field(..., description="Stable task identifier.")
    difficulty: Difficulty
    reference_context: str = Field(
        default="",
        description="Synthetic context (invoice, calendar, log, …) shown to the agent.",
    )
    grader_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Deterministic score in [0,1] for the last assistant message.",
    )
    grader_detail: str = Field(default="", description="Short, reproducible grader note.")
    max_steps: int = Field(default=6, ge=1)
    step_limit_reached: bool = False


class SpinalBenchState(State):
    """Session state for loop detection and task tracking."""

    task_id: str = ""
    difficulty: Difficulty = "easy"
    last_reply_norm: str = ""
    terminal: bool = False
    cumulative_grader: float = 0.0
