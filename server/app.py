"""FastAPI application for the Time Travel Rewind environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install 'openenv-core[core]'"
    ) from e

try:
    from ..models import TimetravelAction, TimetravelObservation
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from models import TimetravelAction, TimetravelObservation

from .timetravel_environment import TimetravelEnvironment

app = create_app(
    TimetravelEnvironment,
    TimetravelAction,
    TimetravelObservation,
    env_name="timetravel",
    max_concurrent_envs=64,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
