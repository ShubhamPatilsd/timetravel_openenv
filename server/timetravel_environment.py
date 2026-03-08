"""Time Travel Rewind Environment.

Ports the game logic from the Prime Intellect verifiers environment
(time_travel_rewind.py) into OpenEnv's FastAPI-based server format.

Map: start -> door -> path-1 -> path-2 -> sign

The agent must navigate to `sign`, read the passcode, then branch back
to a point near `door` and open it efficiently. Reward is highest when
the agent solves the puzzle in fewest steps after branching.
"""

from __future__ import annotations

import copy
import json
import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TimetravelAction, TimetravelObservation
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from models import TimetravelAction, TimetravelObservation


PASSCODES = ("AURORA-314", "CINDER-271", "EMBER-907", "NOVA-553")
LOCATIONS = ("start", "door", "path-1", "path-2", "sign")


class TimetravelEnvironment(Environment):
    """Time Travel Rewind environment with full game logic and RL rewards.

    Game:
        Map: start -> door -> path-1 -> path-2 -> sign
        - Door at position 1 requires a passcode.
        - Passcode is revealed only at sign (position 4).
        - Budget limits total actions per episode.
        - Agent can branch(ago, instruction) to rewind state + send message to past self.

    Reward (computed at episode end):
        solved             * 1.0   (1.0 if door unlocked, else 0.0)
        final_path_eff     * 0.5   (1/path_len after last branch, 0 if not solved)
        json_format_score  * 0.2   (fraction of turns with valid JSON)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, budget: int = 12, episode_index: int = 0):
        self._initial_budget = budget
        self._episode_index = episode_index
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._init_game()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_game(self) -> None:
        """Reset all game state for a new episode."""
        self._target_passcode: str = PASSCODES[self._episode_index % len(PASSCODES)]
        self._position: int = 0
        self._budget_remaining: int = self._initial_budget
        self._active_step_count: int = 0
        self._active_timeline_id: int = 0
        self._next_timeline_id: int = 1
        self._read_sign_count: int = 0
        self._num_branches: int = 0
        self._branch_ago_sum: int = 0
        self._wrong_open_attempts: int = 0
        self._steps_since_last_branch: int = 0
        self._final_path_length: int = -1
        self._succeeded: bool = False
        self._abandoned: bool = False
        self._protocol_violations: int = 0
        self._json_total_turns: int = 0
        self._json_valid_turns: int = 0
        self._event_log: List[Dict[str, Any]] = []
        self._checkpoints: List[Dict[str, Any]] = [self._snapshot()]

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "position": self._position,
            "active_step_count": self._active_step_count,
            "read_sign_count": self._read_sign_count,
            "steps_since_last_branch": self._steps_since_last_branch,
        }

    def _consume_action(self) -> None:
        self._budget_remaining -= 1
        self._active_step_count += 1
        self._steps_since_last_branch += 1

    def _is_done(self) -> bool:
        return self._succeeded or self._abandoned or self._budget_remaining <= 0

    def _position_name(self) -> str:
        return LOCATIONS[self._position]

    def _compute_reward(self) -> float:
        solved = 1.0 if self._succeeded else 0.0
        path_eff = 0.0
        if self._succeeded and self._final_path_length > 0:
            path_eff = 1.0 / float(self._final_path_length)
        json_score = 0.0
        if self._json_total_turns > 0:
            json_score = float(self._json_valid_turns) / float(self._json_total_turns)
        return solved * 1.0 + path_eff * 0.5 + json_score * 0.2

    def _parse_json_action(
        self, content: str
    ) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """Parse model output JSON. Returns (thinking, action, args) or None on failure."""
        self._json_total_turns += 1
        text = content.strip()
        if not text:
            self._protocol_violations += 1
            return None
        try:
            payload = json.loads(text)
        except Exception:
            self._protocol_violations += 1
            return None
        if not isinstance(payload, dict):
            self._protocol_violations += 1
            return None
        thinking = payload.get("thinking")
        action = payload.get("action")
        args = payload.get("args", {})
        if not isinstance(thinking, str) or len(thinking.strip()) < 8:
            self._protocol_violations += 1
            return None
        if not isinstance(action, str):
            self._protocol_violations += 1
            return None
        if not isinstance(args, dict):
            self._protocol_violations += 1
            return None
        self._json_valid_turns += 1
        return thinking.strip(), action.strip(), args

    def _build_obs(
        self,
        message: str,
        reward: float,
        done: bool,
        temporal_note: Optional[str] = None,
    ) -> TimetravelObservation:
        self._state.step_count = self._active_step_count
        return TimetravelObservation(
            message=message,
            position=self._position_name(),
            budget_remaining=self._budget_remaining,
            active_timeline_id=self._active_timeline_id,
            num_branches=self._num_branches,
            read_sign_count=self._read_sign_count,
            succeeded=self._succeeded,
            protocol_violations=self._protocol_violations,
            temporal_note=temporal_note,
            done=done,
            reward=reward,
            metadata={
                "active_step_count": self._active_step_count,
                "wrong_open_attempts": self._wrong_open_attempts,
                "final_path_length": self._final_path_length,
                "json_total_turns": self._json_total_turns,
                "json_valid_turns": self._json_valid_turns,
                "event_log": self._event_log[-10:],  # last 10 events only
            },
        )

    def _task_prompt(self) -> str:
        return (
            f"Temporal gate mission. Budget: {self._initial_budget} actions.\n"
            "Map: start -> door -> path-1 -> path-2 -> sign\n"
            "You start at `start`. The door is locked and requires a passcode.\n"
            "The passcode can only be read at `sign`.\n\n"
            "Output exactly one JSON object per turn:\n"
            '{"thinking":"...", "action":"...", "args":{...}}\n\n'
            "Allowed actions:\n"
            "  move_forward  - move one step forward  (args: {})\n"
            "  move_back     - move one step back      (args: {})\n"
            "  read_sign     - read sign at `sign`     (args: {})\n"
            "  open_door     - try passcode at `door`  (args: {\"passcode\":\"...\"})\n"
            "  branch        - rewind timeline          (args: {\"ago\":<int>, \"instruction\":\"...\"})\n"
            "  abandon       - give up                  (args: {})\n\n"
            "Temporal policy:\n"
            "- Treat messages from future selves as high-value evidence.\n"
            "- After receiving actionable future info, pivot immediately.\n"
            "- Prefer shortest-path execution; avoid redundant exploration.\n\n"
            "Intended strategy: reach sign, read passcode, branch back near door, open door."
        )

    # ------------------------------------------------------------------
    # Game actions
    # ------------------------------------------------------------------

    def _move_forward(self) -> str:
        self._consume_action()
        if self._position >= len(LOCATIONS) - 1:
            feedback = "You are already at sign. Cannot move forward."
        else:
            self._position += 1
            here = self._position_name()
            if here == "door":
                feedback = "You moved to door. The lockpad awaits: open_door(passcode)."
            elif here == "sign":
                feedback = "You moved to sign. You can now call read_sign()."
            else:
                feedback = f"You moved forward to {here}."
        self._event_log.append({
            "event": "move", "direction": "forward",
            "timeline_id": self._active_timeline_id,
            "position": self._position_name(),
            "budget_remaining": self._budget_remaining,
        })
        self._checkpoints.append(self._snapshot())
        return feedback

    def _move_back(self) -> str:
        self._consume_action()
        if self._position <= 0:
            feedback = "You are already at start. Cannot move back."
        else:
            self._position -= 1
            feedback = f"You moved back to {self._position_name()}."
        self._event_log.append({
            "event": "move", "direction": "back",
            "timeline_id": self._active_timeline_id,
            "position": self._position_name(),
            "budget_remaining": self._budget_remaining,
        })
        self._checkpoints.append(self._snapshot())
        return feedback

    def _read_sign(self) -> str:
        self._consume_action()
        if self._position_name() != "sign":
            feedback = "No sign here. Move to sign first."
        else:
            self._read_sign_count += 1
            feedback = (
                f"Sign inscription: PASSCODE = {self._target_passcode}. "
                "Carry this back with branch(instruction=...)."
            )
        self._event_log.append({
            "event": "read_sign",
            "timeline_id": self._active_timeline_id,
            "position": self._position_name(),
            "read_sign_count": self._read_sign_count,
            "budget_remaining": self._budget_remaining,
        })
        self._checkpoints.append(self._snapshot())
        return feedback

    def _open_door(self, passcode: str) -> str:
        self._consume_action()
        submitted = passcode.strip()
        if self._position_name() != "door":
            feedback = "No door lock here. Move to door first."
        elif submitted == self._target_passcode:
            self._succeeded = True
            self._final_path_length = self._steps_since_last_branch
            feedback = (
                f"Door unlocked with {submitted}! Mission complete with "
                f"{self._budget_remaining} budget remaining."
            )
        else:
            self._wrong_open_attempts += 1
            feedback = "Incorrect passcode. The door stays locked."
        self._event_log.append({
            "event": "open_door",
            "timeline_id": self._active_timeline_id,
            "submitted": submitted,
            "correct": self._succeeded,
            "budget_remaining": self._budget_remaining,
        })
        self._checkpoints.append(self._snapshot())
        return feedback

    def _branch(self, ago: int, instruction: str) -> Tuple[str, Optional[str]]:
        """Returns (feedback, temporal_note) tuple."""
        current_steps = self._active_step_count
        self._consume_action()

        if ago <= 0:
            self._checkpoints.append(self._snapshot())
            return "Invalid branch: ago must be > 0.", None

        if ago > current_steps:
            self._checkpoints.append(self._snapshot())
            return f"Invalid branch: ago={ago} exceeds active_step_count={current_steps}.", None

        fork_step = current_steps - ago
        checkpoint = copy.deepcopy(self._checkpoints[fork_step])

        old_timeline_id = self._active_timeline_id
        new_timeline_id = self._next_timeline_id
        self._next_timeline_id += 1

        # Restore game state from checkpoint
        self._position = checkpoint["position"]
        self._active_step_count = checkpoint["active_step_count"] + 1
        self._read_sign_count = checkpoint["read_sign_count"]
        self._steps_since_last_branch = 0
        self._checkpoints = self._checkpoints[: fork_step + 1]
        self._active_timeline_id = new_timeline_id

        self._num_branches += 1
        self._branch_ago_sum += ago
        self._event_log.append({
            "event": "branch",
            "from_timeline": old_timeline_id,
            "to_timeline": new_timeline_id,
            "ago": ago,
            "fork_step": fork_step,
            "position": self._position_name(),
            "budget_remaining": self._budget_remaining,
        })

        temporal_note = (instruction.strip() or "No instruction provided.")[:400]
        note = (
            f"Temporal note from future self: {temporal_note}\n"
            f"(You are now at {self._position_name()} on rewound timeline {new_timeline_id}.)"
        )
        feedback = (
            f"Branch executed. Rewound {ago} active steps from timeline {old_timeline_id} "
            f"to timeline {new_timeline_id}. Now at {self._position_name()}."
        )
        self._checkpoints.append(self._snapshot())
        return feedback, note

    def _abandon(self) -> str:
        self._consume_action()
        self._abandoned = True
        self._event_log.append({
            "event": "abandon",
            "timeline_id": self._active_timeline_id,
            "budget_remaining": self._budget_remaining,
        })
        return "Timeline abandoned by agent request."

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> TimetravelObservation:
        """Reset the environment for a new episode."""
        self._episode_index += 1
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._init_game()
        return self._build_obs(self._task_prompt(), 0.0, False)

    def step(self, action: TimetravelAction) -> TimetravelObservation:  # type: ignore[override]
        """Execute one agent action and return the environment response."""
        if self._is_done():
            return self._build_obs(
                "Episode is done. Call reset() to start a new episode.",
                reward=self._compute_reward(),
                done=True,
            )

        parsed = self._parse_json_action(action.content)
        if parsed is None:
            # Protocol violation — still costs nothing, return error message
            feedback = (
                "Invalid JSON action format. Use exactly: "
                '{"thinking":"...","action":"...","args":{...}}'
            )
            done = self._is_done()
            reward = self._compute_reward() if done else 0.0
            return self._build_obs(feedback, reward, done)

        _, act, args = parsed
        temporal_note: Optional[str] = None

        if act == "move_forward":
            feedback = self._move_forward()
        elif act == "move_back":
            feedback = self._move_back()
        elif act == "read_sign":
            feedback = self._read_sign()
        elif act == "open_door":
            passcode = str(args.get("passcode", ""))
            feedback = self._open_door(passcode)
        elif act == "branch":
            raw_ago = args.get("ago", 0)
            ago = int(raw_ago) if isinstance(raw_ago, (int, str)) else 0
            instruction = str(args.get("instruction", ""))
            result = self._branch(ago, instruction)
            feedback, temporal_note = result
        elif act == "abandon":
            feedback = self._abandon()
        else:
            self._protocol_violations += 1
            self._consume_action()
            self._checkpoints.append(self._snapshot())
            feedback = (
                "Invalid action. Use one of: move_forward, move_back, read_sign, "
                "open_door, branch, abandon."
            )

        done = self._is_done()
        reward = self._compute_reward() if done else 0.0
        return self._build_obs(feedback, reward, done, temporal_note=temporal_note)

    @property
    def state(self) -> State:
        return self._state
