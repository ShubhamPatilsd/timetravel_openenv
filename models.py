"""Data models for the Time Travel Rewind environment."""

from typing import Any, Dict, Literal, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class TimetravelAction(Action):
    """Action for the time travel rewind environment.

    The agent outputs a raw JSON string with the schema:
        {"thinking": "...", "action": "...", "args": {...}}

    Allowed actions:
        move_forward  - move one step toward sign (args: {})
        move_back     - move one step back toward start (args: {})
        read_sign     - read passcode at sign (args: {})
        open_door     - try to unlock door (args: {"passcode": "..."})
        branch        - rewind timeline (args: {"ago": <int>, "instruction": "..."})
        abandon       - end episode (args: {})
    """

    content: str = Field(
        default="",
        description="Raw JSON action from the model: {\"thinking\":\"...\",\"action\":\"...\",\"args\":{...}}",
    )


class TimetravelObservation(Observation):
    """Observation from the time travel rewind environment."""

    message: str = Field(
        default="",
        description="Environment feedback text for this step",
    )
    position: str = Field(
        default="start",
        description="Current position in the map: start | door | path-1 | path-2 | sign",
    )
    budget_remaining: int = Field(
        default=12,
        description="Remaining action budget for this episode",
    )
    active_timeline_id: int = Field(
        default=0,
        description="Current timeline ID (increments on each branch)",
    )
    num_branches: int = Field(
        default=0,
        description="Number of branch actions taken so far",
    )
    read_sign_count: int = Field(
        default=0,
        description="How many times the sign has been read",
    )
    succeeded: bool = Field(
        default=False,
        description="Whether the door has been successfully unlocked",
    )
    protocol_violations: int = Field(
        default=0,
        description="Number of invalid JSON / bad action format violations",
    )
    temporal_note: Optional[str] = Field(
        default=None,
        description="Message from a future self (set after a branch action)",
    )
