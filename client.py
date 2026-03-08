"""Client for the Time Travel Rewind environment."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import TimetravelAction, TimetravelObservation


class TimetravelEnv(EnvClient[TimetravelAction, TimetravelObservation, State]):
    """Client for the Time Travel Rewind environment.

    Example::

        with TimetravelEnv(base_url="http://localhost:8000") as env:
            result = env.reset()
            print(result.observation.message)  # task prompt

            # navigate forward
            result = env.step(TimetravelAction(
                content='{"thinking":"move toward sign","action":"move_forward","args":{}}'
            ))
            print(result.observation.message)
            print(result.observation.position)
    """

    def _step_payload(self, action: TimetravelAction) -> Dict:
        return {"content": action.content}

    def _parse_result(self, payload: Dict) -> StepResult[TimetravelObservation]:
        obs_data = payload.get("observation", {})
        observation = TimetravelObservation(
            message=obs_data.get("message", ""),
            position=obs_data.get("position", "start"),
            budget_remaining=obs_data.get("budget_remaining", 0),
            active_timeline_id=obs_data.get("active_timeline_id", 0),
            num_branches=obs_data.get("num_branches", 0),
            read_sign_count=obs_data.get("read_sign_count", 0),
            succeeded=obs_data.get("succeeded", False),
            protocol_violations=obs_data.get("protocol_violations", 0),
            temporal_note=obs_data.get("temporal_note"),
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

    def _json_action(self, thinking: str, action: str, args: dict) -> TimetravelAction:
        import json
        return TimetravelAction(
            content=json.dumps({"thinking": thinking, "action": action, "args": args})
        )

    def move_forward(self, thinking: str = "moving forward") -> StepResult:
        return self.step(self._json_action(thinking, "move_forward", {}))

    def move_back(self, thinking: str = "moving back") -> StepResult:
        return self.step(self._json_action(thinking, "move_back", {}))

    def read_sign(self, thinking: str = "reading the sign") -> StepResult:
        return self.step(self._json_action(thinking, "read_sign", {}))

    def open_door(self, passcode: str, thinking: str = "opening the door") -> StepResult:
        return self.step(self._json_action(thinking, "open_door", {"passcode": passcode}))

    def branch(self, ago: int, instruction: str, thinking: str = "branching") -> StepResult:
        return self.step(
            self._json_action(thinking, "branch", {"ago": ago, "instruction": instruction})
        )

    def abandon(self, thinking: str = "giving up") -> StepResult:
        return self.step(self._json_action(thinking, "abandon", {}))
