"""Composite FastAPI application mounting both TimeTravel and TextWorld environments."""

import sys
from fastapi import FastAPI

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install 'openenv-core[core]'"
    ) from e

# ------------------------------------------------------------------
# TimeTravel imports
# ------------------------------------------------------------------

try:
    from ..models import TimetravelAction, TimetravelObservation
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from models import TimetravelAction, TimetravelObservation

from .timetravel_environment import TimetravelEnvironment

# ------------------------------------------------------------------
# TextWorld imports
# ------------------------------------------------------------------

try:
    from .textworld_models import TextworldAction, TextworldObservation
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from textworld_models import TextworldAction, TextworldObservation

from .textworld_environment import TextworldEnvironment

# ------------------------------------------------------------------
# Sub-applications
# ------------------------------------------------------------------

print("[startup] creating timetravel app...", flush=True)
timetravel_app = create_app(
    TimetravelEnvironment,
    TimetravelAction,
    TimetravelObservation,
    env_name="timetravel",
    max_concurrent_envs=64,
)
print("[startup] timetravel app created", flush=True)

print("[startup] creating textworld app...", flush=True)
textworld_app = create_app(
    TextworldEnvironment,
    TextworldAction,
    TextworldObservation,
    env_name="textworld",
    max_concurrent_envs=32,
)
print("[startup] textworld app created", flush=True)

# ------------------------------------------------------------------
# Composite root application
# ------------------------------------------------------------------

print("[startup] mounting apps...", flush=True)
app = FastAPI(title="TimeTravelOpenEnv")
app.mount("/timetravel", timetravel_app)
app.mount("/textworld", textworld_app)
print("[startup] all apps mounted, ready", flush=True)


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
