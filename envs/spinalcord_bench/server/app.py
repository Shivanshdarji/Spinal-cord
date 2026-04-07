"""FastAPI entrypoint for OpenEnv / Hugging Face Spaces."""

from __future__ import annotations

import os

from openenv.core.env_server import create_app

from spinalcord_bench.models import SpinalBenchAction, SpinalBenchObservation
from spinalcord_bench.server.spinalcord_bench_env import SpinalBenchEnv

app = create_app(
    SpinalBenchEnv,
    SpinalBenchAction,
    SpinalBenchObservation,
    env_name="spinalcord_bench",
)


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", os.environ.get("API_PORT", "7860")))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
