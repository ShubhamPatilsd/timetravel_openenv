"""Client for the TextWorld temporal environment."""

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .textworld_models import TextworldAction, TextworldObservation
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from textworld_models import TextworldAction, TextworldObservation


class TextworldEnv(EnvClient[TextworldAction, TextworldObservation, State]):
    """Client for the TextWorld temporal environment.

    Connects to the ``/textworld`` mount of the OpenEnv server.

    Example::

        with TextworldEnv(base_url="http://localhost:7860/textworld") as env:
            result = env.reset()
            print(result.observation.feedback)

            result = env.step_command("go east")
            print(result.observation.feedback)
            print(result.observation.score)

            result = env.branch(ago=1, instruction="Go east immediately")
            print(result.observation.active_timeline_id)
    """

    def __init__(self, base_url: str = "http://localhost:7860/textworld", **kwargs):
        super().__init__(base_url=base_url, **kwargs)

    # ------------------------------------------------------------------
    # EnvClient protocol
    # ------------------------------------------------------------------

    def _step_payload(self, action: TextworldAction) -> Dict:
        return {
            "kind": action.kind,
            "command": action.command,
            "instruction": action.instruction,
            "ago": action.ago,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TextworldObservation]:
        obs_data = payload.get("observation", {})
        observation = TextworldObservation(
            feedback=obs_data.get("feedback", ""),
            score=obs_data.get("score", 0.0),
            instruction_hint=obs_data.get("instruction_hint", ""),
            remaining_budget=obs_data.get("remaining_budget", 0),
            current_step=obs_data.get("current_step", 0),
            active_timeline_id=obs_data.get("active_timeline_id", "t0"),
            timeline_status=obs_data.get("timeline_status", "active"),
            event_log_size=obs_data.get("event_log_size", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def step_command(self, command: str) -> StepResult[TextworldObservation]:
        """Send a text command to the game engine (kind="step")."""
        return self.step(TextworldAction(kind="step", command=command))

    def branch(
        self, ago: int, instruction: str
    ) -> StepResult[TextworldObservation]:
        """Rewind ``ago`` steps and start a new timeline with the given instruction."""
        return self.step(
            TextworldAction(kind="branch", ago=ago, instruction=instruction)
        )

    def abandon(self) -> StepResult[TextworldObservation]:
        """End the current episode immediately."""
        return self.step(TextworldAction(kind="abandon"))
