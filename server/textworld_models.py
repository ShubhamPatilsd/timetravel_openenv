"""Pydantic models for the TextWorld temporal environment."""

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation


class TextworldAction(Action):
    """Action for the TextWorld temporal environment.

    Supported kinds:
        step    - send a text command to the game engine (e.g. "go east")
        branch  - rewind the timeline N steps and start a new branch
        abandon - end the episode immediately
    """

    kind: Literal["step", "branch", "abandon"] = "step"
    command: str = "look"
    instruction: str = ""
    ago: Optional[int] = None


class TextworldObservation(Observation):
    """Observation from the TextWorld temporal environment.

    Fields ``done``, ``reward``, and ``metadata`` are inherited from the base
    ``Observation`` class and must not be redeclared here.
    """

    feedback: str
    score: float = 0.0
    instruction_hint: str = ""
    remaining_budget: int = 0
    current_step: int = 0
    active_timeline_id: str = "t0"
    timeline_status: str = "active"
    event_log_size: int = 0
