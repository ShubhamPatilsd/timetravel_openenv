---
title: timetravel
emoji: ⏳
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# timetravel — OpenEnv Time Travel Rewind Environment

A temporal-control RL environment where an agent must navigate a path, discover a passcode at the end, and use time-travel (branch) to rewind and open a locked door efficiently.

## Game

```
start -> door -> path-1 -> path-2 -> sign
```

- The door at position `door` is locked and requires a passcode.
- The passcode is only revealed at `sign`.
- Budget is intentionally tight: walking all the way and back is expensive.
- The optimal strategy: reach `sign`, read passcode, `branch` back near `door`, unlock.

## JSON Action Protocol

Each model turn must output exactly one JSON object:

```json
{"thinking": "...", "action": "...", "args": {...}}
```

| Action | Args | Description |
|--------|------|-------------|
| `move_forward` | `{}` | Move one step toward `sign` |
| `move_back` | `{}` | Move one step back toward `start` |
| `read_sign` | `{}` | Read the passcode (must be at `sign`) |
| `open_door` | `{"passcode": "..."}` | Unlock the door (must be at `door`) |
| `branch` | `{"ago": <int>, "instruction": "..."}` | Rewind `ago` steps, pass note to past self |
| `abandon` | `{}` | End episode immediately |

Invalid JSON or schema violations increment `protocol_violations` and return an error message.

## Temporal Mechanics

- `branch(ago, instruction)`: rewinds internal game state `ago` steps. `ago > 0` required.
- After branching, position and read_sign_count are restored to the checkpoint.
- A `temporal_note` field in the observation carries the instruction from future self.
- Budget is **not** restored on branch — it continues from where it was.
- Previous timeline is archived in the event log.

## Reward

```
reward = solved * 1.0 + final_path_efficiency * 0.5 + json_format_score * 0.2
```

- `solved`: 1.0 if door was unlocked, else 0.0
- `final_path_efficiency`: `1 / steps_after_last_branch` on success, else 0.0
- `json_format_score`: `json_valid_turns / json_total_turns`

Reward is returned at the final step (done=True). All intermediate steps return reward=0.0.

## Running

```bash
# Install dependencies
uv sync

# Start server (dev)
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Or via entry point
uv run server
```

## Environment Variables

| Param | Default | Description |
|-------|---------|-------------|
| `budget` | `12` | Max actions per episode |

## Passcodes

Episodes cycle through: `AURORA-314`, `CINDER-271`, `EMBER-907`, `NOVA-553`.
