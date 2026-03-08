"""Local GRPO training against the timetravel OpenEnv server.

Usage:
    # Terminal 1 - start the env server
    uvicorn server.app:app --host 0.0.0.0 --port 7860

    # Terminal 2 - run training
    python train.py
"""

import json
import os
import requests
from datasets import Dataset

# ── Config ─────────────────────────────────────────────────────────────────────
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
BASE_MODEL   = os.getenv("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_SEQ_LEN  = 2048
LORA_RANK    = 16
BUDGET       = 12
GRPO_STEPS   = 200
BATCH_SIZE   = 2
GRAD_ACCUM   = 4
NUM_ROLLOUTS = 4   # generations per prompt for GRPO

# ── Model ──────────────────────────────────────────────────────────────────────
from unsloth import FastLanguageModel, is_bfloat16_supported

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print(f"Loaded {BASE_MODEL} with LoRA rank {LORA_RANK}")

# ── Environment helpers ─────────────────────────────────────────────────────────
def env_reset(sid: str) -> str:
    r = requests.post(f"{ENV_URL}/reset", json={"session_id": sid}, timeout=10)
    r.raise_for_status()
    return r.json()["observation"]["message"]

def env_step(action_json: str, sid: str):
    r = requests.post(f"{ENV_URL}/step",
                      json={"content": action_json, "session_id": sid},
                      timeout=10)
    r.raise_for_status()
    d = r.json()
    obs = d["observation"]
    return obs["message"], obs.get("temporal_note"), d.get("reward", 0.0), d.get("done", False)

# ── Reward function (called by GRPOTrainer per batch) ──────────────────────────
_reward_counter = 0

def timetravel_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    global _reward_counter
    rewards = []
    for prompt, completion in zip(prompts, completions):
        sid = f"train_{_reward_counter}"
        _reward_counter += 1
        try:
            env_reset(sid)
            final_reward = 0.0
            for line in completion.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    _, _, reward, done = env_step(line, sid)
                    if done:
                        final_reward = reward
                        break
        except Exception as e:
            print(f"[reward] episode error: {e}")
            final_reward = 0.0
        rewards.append(final_reward)
    return rewards

# ── Dataset ────────────────────────────────────────────────────────────────────
TASK_PROMPT = (
    f"Temporal gate mission. Budget: {BUDGET} actions.\n"
    "Map: start -> door -> path-1 -> path-2 -> sign\n"
    "You start at `start`. The door is locked and requires a passcode.\n"
    "The passcode can only be read at `sign`.\n\n"
    "Output exactly one JSON object per turn:\n"
    "{\"thinking\":\"...\", \"action\":\"...\", \"args\":{...}}\n\n"
    "Actions: move_forward, move_back, read_sign, open_door, branch, abandon\n"
    "  open_door args: {\"passcode\": \"...\"}\n"
    "  branch args:    {\"ago\": <int>, \"instruction\": \"...\"}\n\n"
    "Strategy: reach sign, read passcode, branch back near door, open door."
)

PASSCODES = ("AURORA-314", "CINDER-271", "EMBER-907", "NOVA-553")
train_dataset = Dataset.from_list([
    {"prompt": TASK_PROMPT, "answer": pc}
    for pc in PASSCODES * 30  # 120 rows
])

# ── GRPO Training ──────────────────────────────────────────────────────────────
from trl import GRPOTrainer, GRPOConfig

FastLanguageModel.for_training(model)

grpo_config = GRPOConfig(
    output_dir="timetravel-grpo-out",
    learning_rate=5e-6,
    num_train_epochs=1,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=GRPO_STEPS,
    max_completion_length=512,
    num_generations=NUM_ROLLOUTS,
    temperature=0.7,
    beta=0.001,
    logging_steps=5,
    save_steps=50,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[timetravel_reward],
    args=grpo_config,
    train_dataset=train_dataset,
)

print("Starting GRPO training...")
trainer.train()

# ── Save ───────────────────────────────────────────────────────────────────────
model.save_pretrained_merged("timetravel-grpo-final", tokenizer, save_method="merged_16bit")
print("Saved to timetravel-grpo-final/")
