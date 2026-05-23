
"""
MedQA-Hindi DPO Pair Generation
Self-play strategy: creates "chosen" (good) and "rejected" (degraded) answer pairs.
Reads from data/processed/ (works with or without translation).
"""

import os
import json
import random
import re
from pathlib import Path

# Configuration
SEED = 42
random.seed(SEED)

# Paths
DATA_DIR = Path(__file__).parent

# Determine input directory: use translated if it has files, else processed
if (DATA_DIR / "translated").exists() and any((DATA_DIR / "translated").glob("*.jsonl")):
    INPUT_DIR = DATA_DIR / "translated"
    print("📁 Using translated data")
else:
    INPUT_DIR = DATA_DIR / "processed"
    print("📁 Using processed data (no translation found)")

OUTPUT_DIR = DATA_DIR / "dpo"
OUTPUT_DIR.mkdir(exist_ok=True)


def degrade_answer(answer, strategy=None):
    """
    Degrade a good answer to create a "rejected" sample.
    """
    if strategy is None:
        strategy = random.choice(["truncate", "vague", "generic", "incomplete"])
    
    if strategy == "truncate":
        if len(answer) < 20:
            return answer + " (जवाब अधूरा है)"
        cut_point = int(len(answer) * random.uniform(0.3, 0.6))
        truncated = answer[:cut_point]
        last_period = truncated.rfind("।")
        last_en_period = truncated.rfind(".")
        last_q = truncated.rfind("?")
        cut_idx = max(last_period, last_en_period, last_q)
        if cut_idx > 10:
            return truncated[:cut_idx+1] + " (जवाब अधूरा है)"
        return truncated + "..."
    
    elif strategy == "vague":
        # Use raw strings for regex patterns
        replacements = [
            (r'\b\d+\s*mg\b', 'कुछ मिलीग्राम'),
            (r'\b\d+\s*ml\b', 'कुछ मिलीलीटर'),
            (r'\b\d+\s*दिन\b', 'कुछ दिन'),
            (r'\b\d+\s*हफ्ते\b', 'कुछ हफ्ते'),
            (r'\b\d+\s*दिनों\b', 'कुछ दिनों'),
            (r'\b\d+\s*घंटे\b', 'कुछ घंटे'),
            (r'\b\d+\s*बार\b', 'कुछ बार'),
            (r'\b\d+%\b', 'कुछ प्रतिशत'),
        ]
        degraded = answer
        for pattern, replacement in replacements:
            degraded = re.sub(pattern, replacement, degraded, count=1)
        return degraded + " (विवरण अस्पष्ट है)"
    
    elif strategy == "generic":
        generics = [
            "इस बारे में अधिक जानकारी के लिए कृपया अपने डॉक्टर से संपर्क करें।",
            "यह एक सामान्य चिकित्सा प्रश्न है। व्यक्तिगत सलाह के लिए विशेषज्ञ से मिलें।",
            "इस समस्या का समाधान विभिन्न कारकों पर निर्भर करता है। डॉक्टर से परामर्श करें।",
            "इस स्थिति के लिए पेशेवर चिकित्सक से मिलना सबसे अच्छा रहेगा।",
        ]
        return random.choice(generics)
    
    elif strategy == "incomplete":
        sentences = re.split(r'(?<=[।.!?])\s+', answer)
        if len(sentences) > 2:
            idx = random.randint(1, len(sentences)-2)
            sentences.pop(idx)
            return " ".join(sentences) + " (महत्वपूर्ण जानकारी गायब है)"
        return answer[:len(answer)//2] + "... (पूर्ण जानकारी नहीं)"
    
    return answer


def create_dpo_pair(sample):
    """Create a DPO pair from a single sample."""
    messages = sample["messages"]
    question = messages[1]["content"]
    chosen_answer = messages[2]["content"]
    
    if len(chosen_answer) < 30:
        return None
    
    strategy = random.choice(["truncate", "vague", "generic", "incomplete"])
    rejected_answer = degrade_answer(chosen_answer, strategy)
    
    if rejected_answer == chosen_answer:
        rejected_answer = degrade_answer(chosen_answer, "generic")
    
    dpo_sample = {
        "prompt": [
            {"role": "system", "content": messages[0]["content"]},
            {"role": "user", "content": question}
        ],
        "chosen": [
            {"role": "assistant", "content": chosen_answer}
        ],
        "rejected": [
            {"role": "assistant", "content": rejected_answer}
        ],
        "metadata": {
            **sample.get("metadata", {}),
            "degradation_strategy": strategy
        }
    }
    
    return dpo_sample


def process_split(input_file, output_file):
    """Process a single dataset split."""
    print(f"\n🔄 Processing {input_file.name}...")
    
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    
    dpo_samples = []
    skipped = 0
    for sample in samples:
        dpo_sample = create_dpo_pair(sample)
        if dpo_sample:
            dpo_samples.append(dpo_sample)
        else:
            skipped += 1
    
    with open(output_file, "w", encoding="utf-8") as f:
        for s in dpo_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    print(f"   ✅ Created {len(dpo_samples)} DPO pairs (skipped {skipped} too-short answers)")
    print(f"   💾 Saved to {output_file}")
    
    if dpo_samples:
        sample = dpo_samples[0]
        print(f"\n   📋 Sample DPO pair:")
        print(f"   Q: {sample['prompt'][1]['content'][:80]}...")
        print(f"   Chosen: {sample['chosen'][0]['content'][:80]}...")
        print(f"   Rejected: {sample['rejected'][0]['content'][:80]}...")
        print(f"   Strategy: {sample['metadata']['degradation_strategy']}")


def main():
    print("=" * 60)
    print("🏥 MedQA-Hindi DPO Pair Generation")
    print("=" * 60)
    
    print(f"\n📁 Reading from: {INPUT_DIR}")
    
    if not INPUT_DIR.exists():
        print(f"❌ Input directory not found: {INPUT_DIR}")
        print("   Run: python data/prepare_dataset.py")
        return
    
    input_files = sorted(INPUT_DIR.glob("*.jsonl"))
    if not input_files:
        print(f"❌ No JSONL files found in {INPUT_DIR}")
        return
    
    for input_file in input_files:
        output_file = OUTPUT_DIR / input_file.name
        process_split(input_file, output_file)
    
    print("\n" + "=" * 60)
    print("✅ DPO pair generation complete!")
    print("=" * 60)
    print(f"""
DPO datasets saved to: {OUTPUT_DIR}
Files: train.jsonl, validation.jsonl, test.jsonl

Next step: Upload data/ folder to Colab and run training
    python training/qlora_finetune.py
    """)


if __name__ == "__main__":
    main()
