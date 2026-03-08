"""GRPO trainer for the TextWorld temporal environment.

Same custom loop as train.py (Prime Intellect style) but connects to the
textworld sub-environment at /textworld/ws.

Usage:
    # Terminal 1 — server
    uvicorn server.app:app --host 0.0.0.0 --port 7860

    # Terminal 2 — training
    uv run python train_textworld.py
    uv run python train_textworld.py --print-actions
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

# ── Env URL ───────────────────────────────────────────────────────────────────

ENV_URL = "http://localhost:7860/textworld"

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an agent playing a text-adventure game with time-travel abilities.

Output EXACTLY one JSON per turn. Keep "thinking" under 10 words.

{"thinking":"<brief>","kind":"step","command":"<text command>"}
{"thinking":"<brief>","kind":"branch","ago":<int>,"instruction":"<message>"}
{"thinking":"<brief>","kind":"abandon"}

- "step": send a game command. Valid verbs:
    look, examine <obj>, inventory, take <obj>, drop <obj>
    go north/south/east/west/up/down
    open <obj>, close <obj>
    unlock <obj> with <key>, lock <obj> with <key>
    put <obj> in <container>, insert <obj> into <container>
    Do NOT use "use X on Y" — it is not recognised.
- "branch": rewind ago steps, leaving yourself a hint in "instruction"
- "abandon": give up
"""

JSON_CANDIDATE_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)


_MAX_FEEDBACK_CHARS = 600  # cap per-step feedback to prevent context overflow


def obs_to_text(obs: dict, step_num: int) -> str:
    lines = [
        f"Step {step_num} | Budget remaining: {obs['remaining_budget']} | Score: {obs.get('score', 0.0):.2f}",
        f"Timeline: {obs.get('active_timeline_id', '?')} | Status: {obs.get('timeline_status', '?')}",
    ]
    if obs.get("instruction_hint"):
        lines.append(f"[Message from future self]: {obs['instruction_hint']}")
    feedback = obs.get("feedback", "")
    if feedback:
        # Strip leading ASCII-art banners (lines with only printable non-alpha chars)
        fb_lines = feedback.splitlines()
        fb_lines = [l for l in fb_lines if any(c.isalpha() for c in l)]
        feedback = "\n".join(fb_lines).strip()
        if len(feedback) > _MAX_FEEDBACK_CHARS:
            feedback = feedback[:_MAX_FEEDBACK_CHARS] + "..."
        if feedback:
            lines.append(feedback)
    return "\n".join(lines)


def parse_action(text: str) -> Optional[dict]:
    if not text.strip():
        return None
    for candidate in JSON_CANDIDATE_PATTERN.findall(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "kind" in payload:
            return payload
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "kind" in payload:
            return payload
    except json.JSONDecodeError:
        pass
    return None


def format_action(action: dict) -> str:
    return json.dumps(action, separators=(",", ":"))


def infer_success(obs: dict) -> bool:
    return obs.get("timeline_status") == "done" and float(obs.get("score", 0.0)) > 0.0


# ── WebSocket env client ───────────────────────────────────────────────────────

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(__file__))
from server.textworld_models import TextworldAction, TextworldObservation
from openenv.core import EnvClient
from openenv.core.client_types import StepResult


_VALID_ACTION_KEYS = {"kind", "command", "instruction", "ago"}


class TextworldEnv(EnvClient):
    def _step_payload(self, action):
        # Strip extra keys (e.g. "thinking") — server rejects unknown fields
        return {k: v for k, v in action.items() if k in _VALID_ACTION_KEYS}

    def _parse_result(self, payload):
        obs_data = payload.get("observation", {})
        observation = TextworldObservation(
            feedback=obs_data.get("feedback", ""),
            score=float(obs_data.get("score", 0.0)),
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

    def _parse_state(self, payload):
        from openenv.core.env_server.types import State
        return State(episode_id=payload.get("episode_id"), step_count=payload.get("step_count", 0))


def env_reset(env) -> dict:
    result = env.reset()
    obs = result.observation.model_dump()
    obs["done"] = result.done
    obs["reward"] = float(result.reward or 0.0)
    return obs


def env_step(env, action: dict) -> dict:
    result = env.step(action)
    obs = result.observation.model_dump()
    obs["done"] = result.done
    obs["reward"] = float(result.reward or 0.0)
    return obs


# ── Core training functions (same structure as train.py) ──────────────────────

def _apply_template(tokenizer, messages, device):
    out = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    if hasattr(out, "input_ids"):
        out = out.input_ids
    return out.to(device)


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
    import torch

    # Keep system prompt + at most this many recent user/assistant pairs
    _MAX_HISTORY_TURNS = 6

    with TextworldEnv(base_url=ENV_URL) as env:
        obs = env_reset(env)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        transitions: list[tuple] = []
        episode_score = 0.0

        model.eval()
        with torch.inference_mode():
            for step in range(max_episode_steps):
                # Cap history: system prompt + last _MAX_HISTORY_TURNS pairs
                if len(messages) > 1 + _MAX_HISTORY_TURNS * 2:
                    messages = messages[:1] + messages[-(  _MAX_HISTORY_TURNS * 2):]

                messages.append({"role": "user", "content": obs_to_text(obs, step + 1)})
                prompt_ids = _apply_template(tokenizer, messages, model.device)
                action_ids = _generate_until_valid_json_action(
                    model,
                    tokenizer,
                    prompt_ids,
                    max_total_new_tokens=generation_max_new_tokens,
                    chunk_new_tokens=min(64, generation_max_new_tokens),
                    temperature=temperature,
                    do_sample=True,
                )
                action_text = tokenizer.decode(action_ids, skip_special_tokens=True).strip()
                # Strip Qwen3 thinking block residue (</think> and anything before it)
                if "</think>" in action_text:
                    action_text = action_text.split("</think>")[-1].strip()
                action = parse_action(action_text)
                if action is None:
                    action = {"thinking": "invalid", "kind": "abandon"}

                # Validate branch ago against actual env step count to avoid server error
                if action.get("kind") == "branch":
                    env_current_step = obs.get("current_step", 0)
                    ago = action.get("ago", 0)
                    try:
                        ago = int(ago)
                    except (TypeError, ValueError):
                        ago = 0
                    if ago <= 0 or ago > env_current_step:
                        # Invalid ago — downgrade to a no-op step
                        action = {"thinking": "invalid branch", "kind": "step", "command": "look"}

                obs = env_step(env, action)
                episode_score = float(obs.get("score", 0.0))

                if debug_prefix is not None:
                    print(
                        f"\n{debug_prefix} ── step {step+1}/{max_episode_steps} ──────────────",
                        flush=True,
                    )
                    print(f"{debug_prefix} ENV      → {messages[-1]['content']!r}", flush=True)
                    print(f"{debug_prefix} MODEL    → {action_text!r}", flush=True)
                    print(
                        f"{debug_prefix} RESULT   score={obs.get('score', 0.0):.3f} "
                        f"reward={obs['reward']:.3f} budget={obs.get('remaining_budget')} "
                        f"timeline={obs.get('active_timeline_id')} "
                        f"done={obs['done']} tokens={len(action_ids)}",
                        flush=True,
                    )
                    if obs.get("instruction_hint"):
                        print(f"{debug_prefix} HINT     → {obs['instruction_hint']!r}", flush=True)
                    if obs.get("feedback"):
                        print(f"{debug_prefix} FEEDBACK → {obs['feedback']!r}", flush=True)
                    if debug_full_tokens:
                        full_decoded = tokenizer.decode(action_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                        print(f"{debug_prefix} token_ids={action_ids.tolist()}", flush=True)
                        print(f"{debug_prefix} full_decoded={full_decoded!r}", flush=True)

                messages.append({"role": "assistant", "content": format_action(action)})
                transitions.append((prompt_ids[0].cpu(), action_ids.cpu(), float(obs["reward"])))

                if action.get("kind") == "branch":
                    ago = action.get("ago", 0)
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
        if debug_prefix is not None:
            print(
                f"{debug_prefix} ── EPISODE END score={episode_score:.3f} "
                f"success={infer_success(obs)} ──",
                flush=True,
            )
        return transitions, infer_success(obs)


def compute_episode_return(transitions) -> float:
    return sum(reward for _, _, reward in transitions)


def policy_loss(model, prompt_ids, action_ids, advantage: float):
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
    total_score = 0.0
    branch_used = 0

    model.eval()
    with torch.inference_mode():
        for _ in range(num_episodes):
            with TextworldEnv(base_url=ENV_URL) as env:
                obs = env_reset(env)
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                used_branch = False

                for step in range(max_episode_steps):
                    messages.append({"role": "user", "content": obs_to_text(obs, step + 1)})
                    prompt_ids = _apply_template(tokenizer, messages, model.device)
                    action_ids = _generate_until_valid_json_action(
                        model, tokenizer, prompt_ids,
                        max_total_new_tokens=max_new_tokens,
                        chunk_new_tokens=min(32, max_new_tokens),
                        temperature=0.0,
                        do_sample=False,
                    )
                    action_text = tokenizer.decode(action_ids, skip_special_tokens=True).strip()
                    action = parse_action(action_text)
                    if action is None:
                        action = {"thinking": "invalid", "kind": "abandon"}

                    messages.append({"role": "assistant", "content": format_action(action)})
                    obs = env_step(env, action)

                    if action.get("kind") == "branch":
                        used_branch = True
                    if obs["done"]:
                        break

                successes += int(infer_success(obs))
                total_score += float(obs.get("score", 0.0))
                branch_used += int(used_branch)

    episodes = num_episodes
    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "avg_score": total_score / episodes,
        "branch_rate": branch_used / episodes,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Unsloth model on TextWorld temporal env")
    parser.add_argument("--model-name", default="unsloth/Qwen3-14B-unsloth-bnb-4bit")
    parser.add_argument("--output-dir", default="runs/textworld")
    parser.add_argument("--env-url", default="http://localhost:7860/textworld")

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

    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--generation-max-new-tokens", type=int, default=128)

    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--print-actions", action="store_true")
    parser.add_argument("--print-full-tokens", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--wandb-project", default="textworld-timetravel")
    parser.add_argument("--wandb-run-name", default="qwen3-14b-textworld")

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
    print("Starting TextWorld training...", flush=True)
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

                print(
                    f"[grpo] step={train_step} ep={episode_idx} "
                    f"returns={[round(r,3) for r in returns]} "
                    f"mean={mean_ret:.3f} std={std_ret:.6f}",
                    flush=True,
                )
                step_returns.extend(returns)

                for (transitions, _, _), advantage in zip(group_rollouts, advantages):
                    for prompt_ids, action_ids, _ in transitions:
                        if len(action_ids) == 0:
                            continue
                        if len(prompt_ids) + len(action_ids) > args.max_seq_length:
                            continue  # skip sequences that exceed model max length
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
                    f"[update] step={train_step} optimizer.step() "
                    f"transitions={num_transitions} grad_norm={float(grad_norm):.4f} "
                    f"step_ms={step_ms:.1f}",
                    flush=True,
                )
            else:
                print(f"[update] step={train_step} skipped (no transitions)", flush=True)

            denom = args.episodes_per_step * args.num_generations
            success_rate = step_successes / denom
            avg_return = sum(step_returns) / len(step_returns)
            loss_value = total_loss_value / max(num_transitions, 1)

            # ── Clear reward logging every step ───────────────────────────────
            print(
                f"\n{'='*60}\n"
                f"STEP {train_step:4d}/{args.num_train_steps} | "
                f"success={success_rate:.0%} | "
                f"avg_return={avg_return:.4f} | "
                f"loss={loss_value:.4f}\n"
                f"{'='*60}",
                flush=True,
            )

            row = {
                "step": train_step,
                "success_rate": success_rate,
                "avg_return": avg_return,
                "loss": loss_value,
                "num_transitions": num_transitions,
            }

            if args.eval_every > 0 and (train_step % args.eval_every == 0):
                print(f"[eval] Running {args.eval_episodes} eval episodes...", flush=True)
                eval_metrics = evaluate_model(
                    model, tokenizer,
                    num_episodes=args.eval_episodes,
                    max_episode_steps=args.max_episode_steps,
                    max_new_tokens=args.generation_max_new_tokens,
                )
                row.update({f"eval_{k}": v for k, v in eval_metrics.items()})
                print(
                    f"[eval] success_rate={eval_metrics['success_rate']:.0%} "
                    f"avg_score={eval_metrics['avg_score']:.3f} "
                    f"branch_rate={eval_metrics['branch_rate']:.0%}",
                    flush=True,
                )

            mf.write(json.dumps(row) + "\n")
            mf.flush()

            if use_wandb:
                import wandb
                wandb.log(row, step=train_step)

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
