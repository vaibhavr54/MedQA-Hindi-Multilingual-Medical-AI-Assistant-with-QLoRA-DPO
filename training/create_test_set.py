#!/usr/bin/env python3
"""
Creates data/translated/test.jsonl from DPO validation set.
Uses 200 held-out samples — never seen during QLoRA or DPO training.

DPO format:
  prompt  → list of messages [{role:system,...}, {role:user,...}]
  chosen  → list with one message [{role:assistant, content:...}]
  rejected→ same
"""

import json
import random
from pathlib import Path

# Paths — run from medqa-hindi/ root
VAL_FILE  = Path("D:/B.TECH/Projects/medqa-hindi/data/dpo/validation.jsonl")
OUT_DIR   = Path("data/translated")
OUT_FILE  = OUT_DIR / "test.jsonl"
N_SAMPLES = 200

# Load DPO validation
samples = []
with open(VAL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(json.loads(line))

print(f"Loaded {len(samples)} validation samples")

# Shuffle and take N
random.seed(42)
random.shuffle(samples)
test_samples = samples[:N_SAMPLES]

def extract_text(field):
    """
    Extract plain string from either:
      - a plain string → return as-is
      - a list of message dicts → return content of last message
    """
    if isinstance(field, str):
        return field
    if isinstance(field, list) and field:
        return field[-1].get("content", "")
    return ""

def extract_user(prompt_field):
    """Extract user question from prompt message list."""
    if isinstance(prompt_field, list):
        for msg in prompt_field:
            if msg.get("role") == "user":
                return msg.get("content", "")
        # fallback: last message
        return prompt_field[-1].get("content", "")
    return str(prompt_field)

def extract_system(prompt_field):
    """Extract system prompt from prompt message list."""
    if isinstance(prompt_field, list):
        for msg in prompt_field:
            if msg.get("role") == "system":
                return msg.get("content", "")
    return "You are a helpful medical assistant."

# Convert DPO format → eval messages format
OUT_DIR.mkdir(parents=True, exist_ok=True)

skipped = 0
written = 0
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for s in test_samples:
        system   = extract_system(s["prompt"])
        question = extract_user(s["prompt"])
        answer   = extract_text(s["chosen"])

        if not question or not answer:
            skipped += 1
            continue

        sample = {
            "messages": [
                {"role": "system",    "content": system},
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        written += 1

print(f"✅ Created {OUT_FILE} with {written} samples ({skipped} skipped)")
print(f"\nSample:")
print(f"  Q: {test_samples[0]['prompt'][-1]['content'][:120]}")