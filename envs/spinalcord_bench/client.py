"""WebSocket client for SpinalCord Bench (persistent session)."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from spinalcord_bench.models import SpinalBenchAction, SpinalBenchObservation, SpinalBenchState


class SpinalBenchClient(EnvClient[SpinalBenchAction, SpinalBenchObservation, SpinalBenchState]):
    def _step_payload(self, action: SpinalBenchAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SpinalBenchObservation]:
        obs = SpinalBenchObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SpinalBenchState:
        return SpinalBenchState(**payload)
