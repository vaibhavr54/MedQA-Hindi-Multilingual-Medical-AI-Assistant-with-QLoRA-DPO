import os
import json
import random
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login

# Configuration
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
random.seed(SEED)

# Paths
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)


def safe_strip(value):
    """Safely strip whitespace, handling None."""
    if value is None:
        return ""
    return str(value).strip()


def load_medquad():
    """Load MedQuAD dataset from HuggingFace."""
    print("📥 Loading MedQuAD (lavita/MedQuAD)...")
    try:
        ds = load_dataset("lavita/MedQuAD", split="train")
        print(f"   Loaded {len(ds)} MedQuAD samples")
        return ds
    except Exception as e:
        print(f"   ⚠️  Failed to load MedQuAD: {e}")
        return None


def load_medmcqa_indic():
    """Load MedMCQA-Indic Hindi subset — handles 'test' split only."""
    print("📥 Loading MedMCQA-Indic (ekacare/MedMCQA-Indic)...")
    try:
        # Try 'train' first, fallback to 'test'
        try:
            ds = load_dataset("ekacare/MedMCQA-Indic", "hi", split="train")
        except Exception:
            print("   ℹ️  'train' split not found, trying 'test' split...")
            ds = load_dataset("ekacare/MedMCQA-Indic", "hi", split="test")
        print(f"   Loaded {len(ds)} MedMCQA-Indic samples")
        return ds
    except Exception as e:
        print(f"   ⚠️  Failed to load MedMCQA-Indic: {e}")
        return None


def load_himed():
    """Load HiMed dataset — requires specific config name."""
    print("📥 Loading HiMed (FreedomIntelligence/HiMed)...")
    configs_to_try = [
        "himed_trad_corpus",
        "himed_west_corpus",
        "himed_trad_bench",
        "himed_west_bench"
    ]

    for config in configs_to_try:
        try:
            print(f"   Trying config: '{config}'...")
            ds = load_dataset("FreedomIntelligence/HiMed", config, split="train")
            print(f"   Loaded {len(ds)} HiMed samples (config: {config})")
            return ds
        except Exception as e:
            print(f"   Config '{config}' failed: {e}")
            continue

    print("   ⚠️  All HiMed configs failed. Skipping.")
    return None


def format_medquad_sample(sample):
    """Convert MedQuAD to unified format with safe field access."""
    return {
        "question": safe_strip(sample.get("question")),
        "answer": safe_strip(sample.get("answer")),
        "source": "medquad",
        "language": "en",
        "category": safe_strip(sample.get("focus_area", "general")),
    }


def format_medmcqa_sample(sample):
    """Convert MedMCQA to unified format with safe field access."""
    question = safe_strip(sample.get("question"))
    options = sample.get("options", {})
    answer_idx = sample.get("answer", "")

    # Build answer text from options
    answer_text = ""
    if isinstance(options, dict) and answer_idx in options:
        answer_text = safe_strip(options[answer_idx])
    elif isinstance(options, list) and str(answer_idx).isdigit():
        idx = int(answer_idx)
        if 0 <= idx < len(options):
            answer_text = safe_strip(options[idx])

    # Fallback: if no answer text, use answer index as hint
    if not answer_text and answer_idx is not None:
        answer_text = f"Option {answer_idx}"

    return {
        "question": question,
        "answer": answer_text,
        "source": "medmcqa_indic",
        "language": "hi",
        "category": safe_strip(sample.get("subject", "general")),
    }


def format_himed_sample(sample):
    """Convert HiMed to unified format with safe field access."""
    return {
        "question": safe_strip(sample.get("question")),
        "answer": safe_strip(sample.get("answer")),
        "source": "himed",
        "language": "hi",
        "category": safe_strip(sample.get("category", "general")),
    }


def create_chat_format(samples):
    """Convert samples to chat template format for Qwen2.5."""
    formatted = []
    for s in samples:
        chat = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are MedQA-Hindi, an AI medical assistant. Provide accurate, helpful medical information in Hindi or English. Always include a medical disclaimer that this is for educational purposes only and not a substitute for professional medical advice."
                },
                {
                    "role": "user",
                    "content": s["question"]
                },
                {
                    "role": "assistant",
                    "content": s["answer"]
                }
            ],
            "metadata": {
                "source": s["source"],
                "language": s["language"],
                "category": s["category"]
            }
        }
        formatted.append(chat)
    return formatted


def split_dataset(samples):
    """Split into train/val/test."""
    random.shuffle(samples)
    n = len(samples)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    return {
        "train": samples[:train_end],
        "validation": samples[train_end:val_end],
        "test": samples[val_end:]
    }


def save_jsonl(data, filepath):
    """Save data as JSONL."""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"   💾 Saved {len(data)} samples to {filepath}")


def main():
    print("=" * 60)
    print("🏥 MedQA-Hindi Data Preparation")
    print("=" * 60)

    all_samples = []

    # Load English dataset (MedQuAD)
    medquad = load_medquad()
    if medquad:
        medquad_samples = [format_medquad_sample(s) for s in medquad]
        all_samples.extend(medquad_samples)
        print(f"   ✅ MedQuAD: {len(medquad_samples)} samples\n")

    # Load Hindi datasets
    medmcqa = load_medmcqa_indic()
    if medmcqa:
        medmcqa_samples = [format_medmcqa_sample(s) for s in medmcqa]
        all_samples.extend(medmcqa_samples)
        print(f"   ✅ MedMCQA-Indic: {len(medmcqa_samples)} samples\n")

    himed = load_himed()
    if himed:
        himed_samples = [format_himed_sample(s) for s in himed]
        all_samples.extend(himed_samples)
        print(f"   ✅ HiMed: {len(himed_samples)} samples\n")

    if not all_samples:
        print("❌ No datasets loaded. Exiting.")
        return

    print(f"📊 Total raw samples: {len(all_samples)}")

    # Filter empty Q/A with safe checking
    all_samples = [
        s for s in all_samples 
        if s["question"] and s["answer"] and len(s["question"]) > 5 and len(s["answer"]) > 5
    ]
    print(f"📊 After filtering empty/short Q/A: {len(all_samples)}")

    # Language distribution
    lang_counts = {}
    for s in all_samples:
        lang = s["language"]
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    print(f"📊 Language distribution: {lang_counts}")

    # Convert to chat format
    print("\n🔄 Converting to chat template format...")
    chat_samples = create_chat_format(all_samples)

    # Split
    print("\n✂️  Splitting into train/validation/test...")
    splits = split_dataset(chat_samples)

    # Save
    print("\n💾 Saving datasets...")
    for split_name, split_data in splits.items():
        filepath = OUTPUT_DIR / f"{split_name}.jsonl"
        save_jsonl(split_data, filepath)

    # Save metadata
    metadata = {
        "total_samples": len(chat_samples),
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "test": len(splits["test"]),
        "sources": list(set(s["metadata"]["source"] for s in chat_samples)),
        "languages": {k: v for k, v in lang_counts.items()},
    }
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Metadata saved to {OUTPUT_DIR / 'metadata.json'}")
    print("\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print("=" * 60)
    print(f"""
Next steps:
  1. Translate English samples: python data/translation.py
  2. Generate DPO pairs: python data/create_dpo_pairs.py
  3. Start training: python training/qlora_finetune.py
    """)


if __name__ == "__main__":
    main()