"""TextWorld temporal environment for the OpenEnv server.

Ports TemporalTextWorldEnv from textworld_temporal.py into the OpenEnv
Environment interface, adding:
  - _StubBackend for stub/offline testing (no game file required)
  - Three bug fixes over the original implementation
  - TextworldObservation return type from _observation()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from .textworld_models import TextworldAction, TextworldObservation
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from server.textworld_models import TextworldAction, TextworldObservation


class TextworldEnvironment(Environment):
    """Temporal TextWorld environment with reverse-only timeline branching.

    Rewind is implemented via deterministic replay from reset through preserved
    action history, so no native snapshot support is required from the backend.

    If ``game_file`` is None, a built-in _StubBackend is used that provides a
    minimal two-room navigation game for testing.

    Bug fixes versus the original TemporalTextWorldEnv:
        1. On branch, old timeline status is set to "abandoned" (not "paused").
        2. ValueError is raised when ``instruction.strip()`` is empty on branch.
        3. _replay_active_timeline() is wrapped in try/except that reverts
           ``_active_timeline_id`` on failure.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ------------------------------------------------------------------
    # Stub backend (used when game_file is None)
    # ------------------------------------------------------------------

    class _StubBackend:
        """Minimal two-room navigation game for offline testing.

        Rooms: start_room -> end_room via "go east".
        Reaching end_room yields score 1.0 and done=True.
        """

        _ROOMS = ("start_room", "end_room")

        def __init__(self, seed: int = 0) -> None:
            self._room_idx: int = 0
            self._seed = seed

        def reset(self, seed: Optional[int] = None) -> tuple[str, float, bool, Dict[str, Any]]:
            self._room_idx = 0
            return (
                "You are in the start_room. There is an exit to the east.",
                0.0,
                False,
                {},
            )

        def step(self, command: str) -> tuple[str, float, bool, Dict[str, Any]]:
            cmd = command.strip().lower()
            if self._room_idx == 0 and cmd == "go east":
                self._room_idx = 1
                return (
                    "You move east and arrive in the end_room. Well done!",
                    1.0,
                    True,
                    {},
                )
            if self._room_idx == 1:
                return (
                    "You are already in the end_room. There is nothing more to do.",
                    1.0,
                    True,
                    {},
                )
            return (
                f"You try '{command}' but nothing happens. You remain in the start_room.",
                0.0,
                False,
                {},
            )

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        budget: int = 50,
        game_file: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        self._budget = budget
        self._game_file = game_file
        self._default_seed = seed

        # Runtime state – initialised properly in reset()
        self._backend: Any = None
        self._episode_seed: int = seed
        self._timeline_counter: int = 0
        self._event_counter: int = 0
        self._total_cost: int = 0
        self._active_timeline_id: str = ""
        self._timelines: Dict[str, Dict[str, Any]] = {}
        self._meta_events: List[Dict[str, Any]] = []
        self._episode_done: bool = False
        self._last_branch_event: Optional[Dict[str, Any]] = None
        self._state: State = State(episode_id=str(uuid4()), step_count=0)

    # ------------------------------------------------------------------
    # Backend factory
    # ------------------------------------------------------------------

    def _make_backend(self) -> Any:
        if self._game_file is None:
            return self._StubBackend(seed=self._episode_seed)
        try:
            import textworld  # type: ignore
        except Exception as exc:
            raise ImportError(
                "textworld is required for game_file mode. "
                "Install with `pip install textworld`."
            ) from exc

        env = textworld.start(self._game_file)
        return _NativeAdapter(env)

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> TextworldObservation:
        """Reset the environment for a new episode."""
        self._episode_seed = self._default_seed
        self._timeline_counter = 1
        self._event_counter = 0
        self._total_cost = 0
        self._episode_done = False
        self._meta_events = []
        self._last_branch_event = None
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self._backend = self._make_backend()
        feedback, score, done, info = self._backend.reset(seed=self._episode_seed)

        self._active_timeline_id = f"t{self._timeline_counter}"
        self._timelines = {
            self._active_timeline_id: {
                "timeline_id": self._active_timeline_id,
                "source_timeline_id": None,
                "status": "active",
                "instruction_hint": "",
                "actions": [],
                "states": [
                    {
                        "feedback": feedback,
                        "score": score,
                        "done": done,
                        "info": info,
                    }
                ],
                "archived_future": [],
            }
        }

        self._log_event(
            event_type="reset",
            reward=0.0,
            done=done,
        )
        return self._observation(reward=0.0, done=done)

    def step(self, action: TextworldAction) -> TextworldObservation:  # type: ignore[override]
        """Execute one temporal action and return an observation."""
        if self._episode_done:
            return self._observation(reward=0.0, done=True)

        self._total_cost += 1

        if action.kind == "branch":
            reward, done = self._handle_branch(action)
        elif action.kind == "abandon":
            reward, done = self._handle_abandon()
        else:
            reward, done = self._handle_step(action)

        if self.remaining_budget <= 0 and not done:
            self._episode_done = True
            done = True

        self._state.step_count = self._current_step()
        return self._observation(reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------------

    @property
    def remaining_budget(self) -> int:
        return max(self._budget - self._total_cost, 0)

    # ------------------------------------------------------------------
    # Internal action handlers
    # ------------------------------------------------------------------

    def _handle_step(self, action: TextworldAction) -> tuple[float, bool]:
        if self._backend is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        command = action.command.strip()
        if not command:
            raise ValueError("step action requires a non-empty command")

        prev_score = float(self._current_state().get("score", 0.0))
        feedback, score, done, backend_info = self._backend.step(command)

        shaped_reward = score - prev_score

        if done:
            self._episode_done = True
            self._active_timeline()["status"] = "done"

        self._active_timeline()["actions"].append(command)
        self._active_timeline()["states"].append(
            {
                "feedback": feedback,
                "score": score,
                "done": done,
                "info": backend_info,
            }
        )

        self._log_event(event_type="step", reward=shaped_reward, done=done, command=command)
        return shaped_reward, done

    def _handle_branch(self, action: TextworldAction) -> tuple[float, bool]:
        """Branch: rewind ago steps and start a new timeline.

        Bug fixes applied here:
          1. Old timeline status set to "abandoned" (not "paused").
          2. ValueError if instruction is empty.
          3. _replay_active_timeline() wrapped in try/except with revert on failure.
        """
        if action.ago is None:
            raise ValueError("branch action requires ago")
        if action.ago <= 0:
            raise ValueError("branch requires ago > 0")
        if action.ago > self._current_step():
            raise ValueError("branch ago cannot exceed current_step")

        # Bug fix 2: reject empty instruction
        instruction = (action.instruction or "").strip()
        if not instruction:
            raise ValueError("branch action requires a non-empty instruction")

        old_timeline_id = self._active_timeline_id
        old_timeline = self._active_timeline()
        source_step = self._current_step()
        rewind_to_step = source_step - action.ago

        # Bug fix 1: set status to "abandoned" not "paused"
        old_timeline["status"] = "abandoned"
        old_timeline["archived_future"].append(
            {
                "from_step": rewind_to_step,
                "actions": old_timeline["actions"][rewind_to_step:],
                "states": old_timeline["states"][rewind_to_step + 1:],
            }
        )

        self._timeline_counter += 1
        new_timeline_id = f"t{self._timeline_counter}"
        preserved_actions = list(old_timeline["actions"][:rewind_to_step])

        self._timelines[new_timeline_id] = {
            "timeline_id": new_timeline_id,
            "source_timeline_id": old_timeline_id,
            "status": "active",
            "instruction_hint": instruction,
            "actions": preserved_actions,
            "states": [],
            "archived_future": [],
        }

        # Bug fix 3: revert active timeline id if replay fails
        prev_active = self._active_timeline_id
        self._active_timeline_id = new_timeline_id
        try:
            self._replay_active_timeline()
        except Exception:
            self._active_timeline_id = prev_active
            raise

        self._last_branch_event = {
            "from_timeline_id": old_timeline_id,
            "to_timeline_id": new_timeline_id,
            "source_step": source_step,
            "rewind_to_step": rewind_to_step,
            "ago": action.ago,
            "instruction": instruction,
        }

        self._log_event(
            event_type="branch",
            reward=0.0,
            done=False,
            instruction=instruction,
            ago=action.ago,
            source_timeline_id=old_timeline_id,
        )
        return 0.0, False

    def _handle_abandon(self) -> tuple[float, bool]:
        self._active_timeline()["status"] = "abandoned"
        self._episode_done = True
        self._log_event(event_type="abandon", reward=0.0, done=True)
        return 0.0, True

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def _replay_active_timeline(self) -> None:
        """Reset backend and replay preserved actions to reconstruct state."""
        self._backend = self._make_backend()
        feedback, score, done, info = self._backend.reset(seed=self._episode_seed)

        timeline = self._active_timeline()
        commands = list(timeline["actions"])
        timeline["states"] = [
            {
                "feedback": feedback,
                "score": score,
                "done": done,
                "info": info,
            }
        ]

        for command in commands:
            feedback, score, done, info = self._backend.step(command)
            timeline["states"].append(
                {
                    "feedback": feedback,
                    "score": score,
                    "done": done,
                    "info": info,
                }
            )

        self._episode_done = bool(timeline["states"][-1]["done"])
        timeline["status"] = "done" if self._episode_done else "active"

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _observation(self, *, reward: float, done: bool) -> TextworldObservation:
        state = self._current_state()
        timeline = self._active_timeline()
        return TextworldObservation(
            feedback=state.get("feedback", ""),
            score=float(state.get("score", 0.0)),
            instruction_hint=timeline.get("instruction_hint", ""),
            remaining_budget=self.remaining_budget,
            current_step=self._current_step(),
            active_timeline_id=self._active_timeline_id,
            timeline_status=timeline["status"],
            event_log_size=len(self._meta_events),
            reward=reward,
            done=done,
            metadata={"last_branch_event": self._last_branch_event},
        )

    # ------------------------------------------------------------------
    # Timeline helpers
    # ------------------------------------------------------------------

    def _active_timeline(self) -> Dict[str, Any]:
        return self._timelines[self._active_timeline_id]

    def _current_step(self) -> int:
        return len(self._active_timeline()["actions"])

    def _current_state(self) -> Dict[str, Any]:
        return self._active_timeline()["states"][-1]

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"event-{self._event_counter}"

    def _log_event(
        self,
        *,
        event_type: str,
        reward: float,
        done: bool,
        command: str = "",
        instruction: str = "",
        ago: Optional[int] = None,
        source_timeline_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        event = {
            "event_id": self._next_event_id(),
            "event_type": event_type,
            "step_index": self._current_step(),
            "timeline_id": self._active_timeline_id,
            "source_timeline_id": source_timeline_id,
            "ago": ago,
            "instruction": instruction,
            "message": command,
            "reward": reward,
            "done": done,
            "status_after_event": self._active_timeline()["status"],
            "remaining_budget": self.remaining_budget,
        }
        self._meta_events.append(event)
        return event


# ---------------------------------------------------------------------------
# Thin adapter around the native TextWorld runtime
# ---------------------------------------------------------------------------

class _NativeAdapter:
    """Wraps a textworld game env so it matches the backend protocol."""

    def __init__(self, env: Any) -> None:
        self._env = env

    def reset(self, seed: Optional[int] = None) -> tuple[str, float, bool, Dict[str, Any]]:
        game_state = self._env.reset()
        feedback = getattr(game_state, "feedback", "")
        score = float(getattr(game_state, "score", 0.0) or 0.0)
        done = bool(getattr(game_state, "game_ended", False))
        return feedback, score, done, {}

    def step(self, command: str) -> tuple[str, float, bool, Dict[str, Any]]:
        game_state, score, done = self._env.step(command)
        feedback = getattr(game_state, "feedback", "")
        return feedback, float(score), bool(done), {}
