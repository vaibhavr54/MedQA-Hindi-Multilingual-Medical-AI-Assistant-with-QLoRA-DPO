#!/usr/bin/env python3
"""
MedQA-Hindi Translation Pipeline
Translates English Q&A pairs to Hindi using IndicTrans2.
"""

import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Configuration
BATCH_SIZE = 8
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
DATA_DIR = Path(__file__).parent
INPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "translated"
OUTPUT_DIR.mkdir(exist_ok=True)

# IndicTrans2 model
MODEL_NAME = "ai4bharat/indic-trans2-en-indic"


class IndicTranslator:
    """Wrapper for IndicTrans2 translation model."""

    def __init__(self):
        print(f"📥 Loading IndicTrans2 model ({MODEL_NAME})...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None
        )
        if DEVICE == "cpu":
            self.model = self.model.to(DEVICE)
        print(f"   ✅ Model loaded on {DEVICE}")

    def translate_batch(self, texts, src_lang="eng_Latn", tgt_lang="hin_Deva"):
        """Translate a batch of texts."""
        if not texts:
            return []

        # Prepare inputs with language tokens
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(DEVICE)

        # Generate translations
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                num_beams=4,
                early_stopping=True
            )

        translations = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        return translations


def translate_dataset(translator, input_file, output_file):
    """Translate all English samples in a dataset file."""
    print(f"\n🔄 Processing {input_file.name}...")

    # Load data
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    # Identify English samples
    en_indices = [
        i for i, s in enumerate(samples) 
        if s["metadata"]["language"] == "en"
    ]

    if not en_indices:
        print(f"   ℹ️  No English samples to translate")
        # Just copy file
        with open(output_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        return

    print(f"   📝 Found {len(en_indices)} English samples to translate")

    # Extract questions and answers
    questions = [samples[i]["messages"][1]["content"] for i in en_indices]
    answers = [samples[i]["messages"][2]["content"] for i in en_indices]

    # Translate in batches
    translated_questions = []
    translated_answers = []

    print("   🔄 Translating questions...")
    for i in tqdm(range(0, len(questions), BATCH_SIZE)):
        batch = questions[i:i+BATCH_SIZE]
        translated_questions.extend(translator.translate_batch(batch))

    print("   🔄 Translating answers...")
    for i in tqdm(range(0, len(answers), BATCH_SIZE)):
        batch = answers[i:i+BATCH_SIZE]
        translated_answers.extend(translator.translate_batch(batch))

    # Update samples
    for idx, orig_idx in enumerate(en_indices):
        samples[orig_idx]["messages"][1]["content"] = translated_questions[idx]
        samples[orig_idx]["messages"][2]["content"] = translated_answers[idx]
        samples[orig_idx]["metadata"]["language"] = "hi"
        samples[orig_idx]["metadata"]["translated"] = True

    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"   ✅ Saved {len(samples)} samples to {output_file}")


def main():
    print("=" * 60)
    print("🏥 MedQA-Hindi Translation Pipeline")
    print("=" * 60)

    # Check for processed files
    if not INPUT_DIR.exists():
        print(f"❌ Input directory not found: {INPUT_DIR}")
        print("   Run: python data/prepare_dataset.py")
        return

    input_files = list(INPUT_DIR.glob("*.jsonl"))
    if not input_files:
        print(f"❌ No JSONL files found in {INPUT_DIR}")
        return

    # Initialize translator
    translator = IndicTranslator()

    # Process each split
    for input_file in sorted(input_files):
        output_file = OUTPUT_DIR / input_file.name
        translate_dataset(translator, input_file, output_file)

    print("\n" + "=" * 60)
    print("✅ Translation complete!")
    print("=" * 60)
    print(f"""
Translated datasets saved to: {OUTPUT_DIR}
Next step: python data/create_dpo_pairs.py
    """)


if __name__ == "__main__":
    main()
