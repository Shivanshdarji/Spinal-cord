"""Root server entrypoint expected by OpenEnv validator."""

from spinalcord_bench.server.app import app as app
from spinalcord_bench.server.app import main as _bench_main

__all__ = ["app", "main"]


def main() -> None:
    _bench_main()


if __name__ == "__main__":
    main()
