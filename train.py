"""GRPO trainer for the Time Travel Rewind OpenEnv environment.

Replicates the Prime Intellect custom GRPO loop from train_reverse_code_door.py
exactly — same loop structure, same policy_loss, same advantage normalisation,
same generate-until-valid-JSON approach — adapted to call the OpenEnv HTTP
server instead of a local env instance.

Usage:
    # Terminal 1
    uvicorn server.app:app --host 0.0.0.0 --port 7860

    # Terminal 2
    uv run python train.py
    uv run python train.py --num-train-steps 300 --episodes-per-step 4 --num-generations 4
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

# ── Env / agent helpers ────────────────────────────────────────────────────────

ENV_URL = "http://localhost:7860/timetravel"

SYSTEM_PROMPT = """You are an agent navigating a time-travel puzzle.

Map: start -> door -> path-1 -> path-2 -> sign

Goal: unlock the door at `door` with the passcode shown at `sign`.
Budget: limited steps total.

The branch command rewinds time by N steps. After the rewind your chat history is erased back to that point — you will NOT remember what you saw. The only thing that survives is the instruction field, which becomes your temporal note. You MUST embed the exact passcode in the instruction, e.g. "Use passcode AURORA-314 at door". After a branch, read your temporal note to get the passcode.

Output exactly one JSON object and nothing else. No explanation outside the JSON.

Valid formats:
  {"thinking":"...","action":"move_forward","args":{}}
  {"thinking":"...","action":"move_back","args":{}}
  {"thinking":"...","action":"read_sign","args":{}}
  {"thinking":"...","action":"open_door","args":{"passcode":"..."}}
  {"thinking":"...","action":"branch","args":{"ago":N,"instruction":"Use passcode X at door"}}
  {"thinking":"...","action":"abandon","args":{}}
"""

JSON_CANDIDATE_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)


def _apply_template(tokenizer, messages, device):
    """Apply chat template and always return a 2D tensor on `device`."""
    out = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    # Transformers ≥5.x returns BatchEncoding; older versions return a raw tensor
    if hasattr(out, "input_ids"):
        out = out.input_ids
    return out.to(device)


def obs_to_text(obs: dict, step_num: int) -> str:
    lines = [
        f"Step {step_num} | Budget remaining: {obs['budget_remaining']}",
        f"Position: {obs['position']}",
    ]
    if obs.get("temporal_note"):
        lines.append(f"Temporal note from future self: {obs['temporal_note']}")
    if obs.get("message"):
        lines.append(obs["message"])
    return "\n".join(lines)


def parse_action(text: str) -> Optional[dict]:
    """Parse model output into an action dict, or None on failure."""
    if not text.strip():
        return None
    for candidate in JSON_CANDIDATE_PATTERN.findall(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "action" in payload:
            return payload
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "action" in payload:
            return payload
    except json.JSONDecodeError:
        return None
    return None


def format_action(action: dict) -> str:
    """Return compact canonical JSON string for chat history."""
    return json.dumps(action, separators=(",", ":"))


def infer_success(obs: dict) -> bool:
    return bool(obs.get("succeeded", False))


ENV_URL = "http://localhost:7860/timetravel"

# ── WebSocket env calls via TimetravelEnv client ───────────────────────────────

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(__file__))
from models import TimetravelAction, TimetravelObservation
from openenv.core import EnvClient
from openenv.core.client_types import StepResult


class TimetravelEnv(EnvClient):
    def _step_payload(self, action):
        return {"content": action}

    def _parse_result(self, payload):
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
        return StepResult(observation=observation, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload):
        from openenv.core.env_server.types import State
        return State(episode_id=payload.get("episode_id"), step_count=payload.get("step_count", 0))


def env_reset(env) -> dict:
    result = env.reset()
    obs = result.observation.model_dump()
    obs["done"] = result.done
    obs["reward"] = float(result.reward or 0.0)
    return obs


def env_step(env, action_json: str) -> dict:
    result = env.step(action_json)
    obs = result.observation.model_dump()
    obs["done"] = result.done
    obs["reward"] = float(result.reward or 0.0)
    return obs


# ── Core training functions (exact Prime Intellect structure) ──────────────────

def _generate_until_valid_json_action(
    model,
    tokenizer,
    prompt_ids,
    *,
    max_total_new_tokens: int,
    chunk_new_tokens: int,
    temperature: float,
    do_sample: bool,
):
    """Generate incrementally until we can parse a full valid JSON action."""
    import torch

    generated = torch.empty(0, dtype=prompt_ids.dtype, device=prompt_ids.device)
    cursor = prompt_ids
    tokens_left = max_total_new_tokens

    while tokens_left > 0:
        step_tokens = min(chunk_new_tokens, tokens_left)
        attention_mask = torch.ones_like(cursor, device=cursor.device)
        out = model.generate(
            cursor,
            attention_mask=attention_mask,
            max_new_tokens=step_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_ids = out[0][cursor.shape[1]:]
        if len(new_ids) == 0:
            break

        generated = torch.cat([generated, new_ids], dim=0)
        cursor = torch.cat([cursor[0], new_ids]).unsqueeze(0)
        tokens_left -= len(new_ids)

        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if parse_action(text) is not None:
            break

    return generated


def collect_episode(
    model,
    tokenizer,
    *,
    max_episode_steps: int,
    generation_max_new_tokens: int,
    temperature: float,
    debug_prefix: str | None = None,
    debug_full_tokens: bool = False,
) -> tuple[list[tuple], bool]:
    """Roll out one episode and return per-step transitions."""
    import torch

    with TimetravelEnv(base_url=ENV_URL) as env:
        obs = env_reset(env)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        transitions: list[tuple] = []

        model.eval()
        with torch.inference_mode():
            for step in range(max_episode_steps):
                messages.append({"role": "user", "content": obs_to_text(obs, step + 1)})
                prompt_ids = _apply_template(tokenizer, messages, model.device)
                action_ids = _generate_until_valid_json_action(
                    model,
                    tokenizer,
                    prompt_ids,
                    max_total_new_tokens=generation_max_new_tokens,
                    chunk_new_tokens=min(32, generation_max_new_tokens),
                    temperature=temperature,
                    do_sample=True,
                )
                action_text = tokenizer.decode(action_ids, skip_special_tokens=True).strip()
                action = parse_action(action_text)
                if action is None:
                    action = {"thinking": "invalid", "action": "abandon", "args": {}}

                obs = env_step(env, format_action(action))

                if debug_prefix is not None:
                    print(
                        f"\n{debug_prefix} ── step {step+1}/{max_episode_steps} ──────────────────",
                        flush=True,
                    )
                    print(f"{debug_prefix} ENV  → {messages[-1]['content']!r}", flush=True)
                    print(f"{debug_prefix} MODEL→ {action_text!r}", flush=True)
                    print(
                        f"{debug_prefix} RESULT pos={obs['position']!r} "
                        f"budget={obs.get('budget_remaining')} "
                        f"reward={obs['reward']:.3f} done={obs['done']} "
                        f"tokens={len(action_ids)}",
                        flush=True,
                    )
                    if obs.get("temporal_note"):
                        print(f"{debug_prefix} TEMPORAL_NOTE={obs['temporal_note']!r}", flush=True)
                    if obs.get("message"):
                        print(f"{debug_prefix} ENV_MSG={obs['message']!r}", flush=True)
                    if debug_full_tokens:
                        full_decoded = tokenizer.decode(
                            action_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        print(f"{debug_prefix} token_ids={action_ids.tolist()}", flush=True)
                        print(f"{debug_prefix} full_decoded={full_decoded!r}", flush=True)

                messages.append({"role": "assistant", "content": format_action(action)})
                transitions.append((prompt_ids[0].cpu(), action_ids.cpu(), float(obs["reward"])))

                if action.get("action") == "branch":
                    ago = action.get("args", {}).get("ago", 0)
                    if isinstance(ago, (int, str)):
                        try:
                            ago = int(ago)
                        except (TypeError, ValueError):
                            ago = 0
                    if ago > 0:
                        keep = 1 + 2 * (step + 1 - ago)
                        messages = messages[:max(keep, 1)]

                if obs["done"]:
                    break

        model.train()
        return transitions, infer_success(obs)


def compute_episode_return(transitions) -> float:
    return sum(reward for _, _, reward in transitions)


def policy_loss(model, prompt_ids, action_ids, advantage: float):
    """Compute mean token NLL over action tokens weighted by advantage."""
    import torch

    input_ids = torch.cat([prompt_ids, action_ids]).unsqueeze(0).to(model.device)
    labels = torch.full_like(input_ids, -100)
    labels[0, len(prompt_ids):] = action_ids

    outputs = model(input_ids=input_ids, labels=labels)
    return outputs.loss * advantage


def evaluate_model(
    model,
    tokenizer,
    *,
    num_episodes: int,
    max_episode_steps: int,
    max_new_tokens: int,
) -> dict:
    import torch

    successes = 0
    branch_used = 0

    model.eval()
    with torch.inference_mode():
        for _ in range(num_episodes):
            with TimetravelEnv(base_url=ENV_URL) as env:
                obs = env_reset(env)
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                used_branch = False

                for step in range(max_episode_steps):
                    messages.append({"role": "user", "content": obs_to_text(obs, step + 1)})
                    prompt_ids = _apply_template(tokenizer, messages, model.device)
                    action_ids = _generate_until_valid_json_action(
                        model,
                        tokenizer,
                        prompt_ids,
                        max_total_new_tokens=max_new_tokens,
                        chunk_new_tokens=min(32, max_new_tokens),
                        temperature=0.0,
                        do_sample=False,
                    )
                    action_text = tokenizer.decode(action_ids, skip_special_tokens=True).strip()
                    action = parse_action(action_text)
                    if action is None:
                        action = {"thinking": "invalid", "action": "abandon", "args": {}}

                    messages.append({"role": "assistant", "content": format_action(action)})
                    obs = env_step(env, format_action(action))

                    if action.get("action") == "branch":
                        used_branch = True

                    if obs["done"]:
                        break

                successes += int(infer_success(obs))
                branch_used += int(used_branch)

    episodes = num_episodes
    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "branch_rate": branch_used / episodes,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Unsloth model on Time Travel Rewind")
    parser.add_argument("--model-name", default="unsloth/Qwen3-14B-unsloth-bnb-4bit")
    parser.add_argument("--output-dir", default="runs/timetravel")
    parser.add_argument("--env-url", default="http://localhost:7860/timetravel")

    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--lora-rank", type=int, default=4)

    parser.add_argument("--num-train-steps", type=int, default=300)
    parser.add_argument("--episodes-per-step", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--max-episode-steps", type=int, default=12)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)

    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--print-actions", action="store_true")
    parser.add_argument("--print-actions-train-steps", type=int, default=3)
    parser.add_argument("--print-full-tokens", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--wandb-project", default="timetravel")
    parser.add_argument("--wandb-run-name", default="qwen3-14b-timetravel")

    args = parser.parse_args()

    global ENV_URL
    ENV_URL = args.env_url

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.optim import AdamW

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError("unsloth is required.") from exc

    # ── W&B ───────────────────────────────────────────────────────────────────
    try:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        use_wandb = True
    except Exception:
        use_wandb = False
        print("W&B not available, logging to metrics.jsonl only.", flush=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_name}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        max_seq_length=args.max_seq_length,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metrics_file = out_dir / "metrics.jsonl"

    # ── Training loop ──────────────────────────────────────────────────────────
    print("Starting training...", flush=True)
    model.train()
    with metrics_file.open("w", encoding="utf-8") as mf:
        for train_step in range(args.num_train_steps):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            step_successes = 0
            step_returns: list[float] = []
            total_loss_value = 0.0
            num_transitions = 0

            for episode_idx in range(args.episodes_per_step):
                group_rollouts = []
                for gen_idx in range(args.num_generations):
                    debug_prefix = None
                    if args.print_actions:
                        debug_prefix = f"[step={train_step}/{args.num_train_steps} ep={episode_idx} gen={gen_idx}]"
                    transitions, success = collect_episode(
                        model,
                        tokenizer,
                        max_episode_steps=args.max_episode_steps,
                        generation_max_new_tokens=args.generation_max_new_tokens,
                        temperature=args.temperature,
                        debug_prefix=debug_prefix,
                        debug_full_tokens=args.print_full_tokens,
                    )
                    ret = compute_episode_return(transitions)
                    group_rollouts.append((transitions, ret, success))
                    step_successes += int(success)

                returns = [r for _, r, _ in group_rollouts]
                mean_ret = sum(returns) / len(returns)
                std_ret = (sum((r - mean_ret) ** 2 for r in returns) / len(returns)) ** 0.5 + 1e-8
                advantages = [(r - mean_ret) / std_ret for r in returns]
                if train_step < args.print_actions_train_steps:
                    print(
                        f"[grpo] train_step={train_step} ep={episode_idx} returns={returns} "
                        f"mean={mean_ret:.3f} std={std_ret:.6f}",
                        flush=True,
                    )
                step_returns.extend(returns)

                for (transitions, _, _), advantage in zip(group_rollouts, advantages):
                    for prompt_ids, action_ids, _ in transitions:
                        if len(action_ids) == 0:
                            continue
                        loss = policy_loss(model, prompt_ids, action_ids, advantage)
                        loss.backward()
                        total_loss_value += float(loss.detach().item())
                        num_transitions += 1

            if num_transitions > 0:
                scale = 1.0 / num_transitions
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
                grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                step_ms = (time.perf_counter() - step_start) * 1000.0
                print(
                    f"[update] train_step={train_step} optimizer.step() transitions={num_transitions} "
                    f"grad_norm={float(grad_norm):.4f} step_ms={step_ms:.1f}",
                    flush=True,
                )
            else:
                print(f"[update] train_step={train_step} skipped (no transitions)", flush=True)

            denom = args.episodes_per_step * args.num_generations
            success_rate = step_successes / denom
            avg_return = sum(step_returns) / len(step_returns)
            loss_value = total_loss_value / max(num_transitions, 1)

            row = {
                "step": train_step,
                "success_rate": success_rate,
                "avg_return": avg_return,
                "loss": loss_value,
                "num_transitions": num_transitions,
            }

            if args.eval_every > 0 and (train_step % args.eval_every == 0):
                eval_metrics = evaluate_model(
                    model,
                    tokenizer,
                    num_episodes=args.eval_episodes,
                    max_episode_steps=args.max_episode_steps,
                    max_new_tokens=args.generation_max_new_tokens,
                )
                row.update({f"eval_{k}": v for k, v in eval_metrics.items()})

            mf.write(json.dumps(row) + "\n")
            mf.flush()

            if use_wandb:
                import wandb
                wandb.log(row, step=train_step)

            if train_step % 10 == 0:
                print(
                    f"step {train_step:4d} | success={success_rate:.0%} | avg_return={avg_return:.3f} | loss={loss_value:.4f}",
                    flush=True,
                )

            if args.save_every > 0 and train_step > 0 and (train_step % args.save_every == 0):
                ckpt_dir = out_dir / f"checkpoint_step_{train_step}"
                model.save_pretrained(str(ckpt_dir))
                tokenizer.save_pretrained(str(ckpt_dir))

    final_dir = out_dir / "final_adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if use_wandb:
        import wandb
        wandb.finish()
    print(f"Done. Saved adapter to: {final_dir}")


if __name__ == "__main__":
    main()
