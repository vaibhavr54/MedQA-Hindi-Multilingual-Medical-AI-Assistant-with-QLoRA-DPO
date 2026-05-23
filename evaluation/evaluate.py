#!/usr/bin/env python3
"""
MedQA-Hindi Evaluation Script
Computes ROUGE-L, BERTScore, and Medical Accuracy for all model variants.

Fixes applied:
  1. BERTScore uses bert-base-multilingual-cased — handles Hindi + English correctly
  2. Base model corrected to Qwen2.5-0.5B-Instruct — matches local training pipeline
  3. load_model() reloads base in fp16 before merge — avoids quantized merge crash
  4. Test data loaded once, passed as argument — not reloaded per model
  5. system_prompt extraction guarded — falls back to default if role != system
  6. argparse for --max_samples, --models, --output flags
  7. model_key field added to results — compare_models.py index bug fixed
  8. Empty prediction guard — replaced with [EMPTY] to prevent metric crash
  9. adapter_path always cast to Path — .exists() safe throughout
"""

import os
import json
import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7

DEFAULT_SYSTEM_PROMPT = (
    "You are MedQA-Hindi, an AI medical assistant. "
    "Answer medical questions accurately in Hindi or English. "
    "Always remind the user to consult a doctor for personal medical decisions."
)

# ── Paths ──────────────────────────────────────────────────────────────────────
EVAL_DIR   = Path(__file__).parent
DATA_DIR   = EVAL_DIR.parent / "data" / "translated"
OUTPUTS_DIR = EVAL_DIR.parent / "outputs"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="MedQA-Hindi Model Evaluation")
    parser.add_argument(
        "--max_samples", type=int, default=100,
        help="Number of test samples to evaluate per model (default: 100)"
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=["qlora", "dpo"],
        default=["qlora", "dpo"],
        help="Which models to evaluate (default: all fine-tuned models, excluding base)"
    )
    parser.add_argument(
        "--test_file", type=str, default=None,
        help="Override path to test.jsonl (default: data/translated/test.jsonl)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output JSON path (default: evaluation/results/evaluation_results.json)"
    )
    return parser.parse_args()


# ── Data ───────────────────────────────────────────────────────────────────────
def load_test_data(test_file: Path, max_samples: int):
    """Load test dataset once — passed to every evaluate_model() call."""
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    samples = samples[:max_samples]
    print(f"📊 Loaded {len(samples)} test samples from {test_file}")
    return samples


def extract_system_prompt(sample: dict) -> str:
    """
    Safely extract system prompt from sample.
    Guard against samples where messages[0] is not the system turn.
    """
    messages = sample.get("messages", [])
    for msg in messages:
        if msg.get("role") == "system":
            return msg["content"]
    return DEFAULT_SYSTEM_PROMPT


def extract_qa(sample: dict):
    """Extract (question, reference_answer) from a sample."""
    messages = sample.get("messages", [])
    question, reference = "", ""
    for msg in messages:
        if msg.get("role") == "user":
            question = msg["content"]
        elif msg.get("role") == "assistant":
            reference = msg["content"]
    return question, reference


# ── Model loading ──────────────────────────────────────────────────────────────
def _bnb_config():
    """4-bit NF4 quantization config for inference on 4GB VRAM."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )


def load_model(model_path: str, adapter_path: Path = None):
    """
    Load model for evaluation.

    If adapter_path is given:
      - Load base in fp16 (NOT quantized) → attach adapter → merge → unload
      FIX: merge_and_unload() crashes on a 4-bit quantized model.
           Merged models are loaded clean; only standalone inference uses 4-bit.

    If no adapter (already merged or base model):
      - Load with 4-bit quantization for VRAM efficiency.
    """
    model_path = str(model_path)

    if adapter_path is not None and Path(adapter_path).exists():
        # Merge path: reload base clean in fp16
        print(f"📥 Loading base for merge: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"   🔌 Attaching adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        print("   🔀 Merging adapter...")
        model = model.merge_and_unload()

    else:
        # Direct load: quantized for memory efficiency
        print(f"📥 Loading model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=_bnb_config(),
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   ✅ Model ready | VRAM: {allocated:.2f}GB / {total:.2f}GB")
    else:
        print("   ✅ Model ready (CPU)")

    return model, tokenizer


# ── Inference ──────────────────────────────────────────────────────────────────
def generate_answer(model, tokenizer, question: str, system_prompt: str) -> str:
    """Generate answer; returns '[EMPTY]' if model produces nothing."""
    # Guard: ensure system_prompt is a plain string
    if isinstance(system_prompt, list):
        system_prompt = " ".join(system_prompt)

    messages = [
        {"role": "system", "content": str(system_prompt)},
        {"role": "user",   "content": str(question)}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    # FIX: guard against empty generation — prevents metric computation crash
    return response if response else "[EMPTY]"


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_rouge_l(predictions: list, references: list) -> float:
    """ROUGE-L F1 averaged over all samples."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return float(np.mean(scores))


def compute_bertscore(predictions: list, references: list) -> float:
    """
    BERTScore F1 using bert-base-multilingual-cased.

    FIX: Original used lang='hi' which loads a Hindi-only model.
    Dataset is bilingual (~92% Hindi, ~8% English).
    Multilingual model handles both scripts correctly.
    """
    _, _, F1 = bert_score(
        predictions,
        references,
        model_type="bert-base-multilingual-cased",
        device=DEVICE,
        verbose=False
    )
    return float(F1.mean().item())


def compute_medical_accuracy(predictions: list, references: list) -> float:
    """
    Keyword-overlap medical accuracy.
    Measures what fraction of medical terms in the reference appear in the prediction.
    Bilingual keyword list covers both Hindi and English terms.
    """
    medical_keywords = [
        # Hindi
        "दवा", "उपचार", "लक्षण", "बीमारी", "संक्रमण", "टीका",
        "रक्त", "हृदय", "फेफड़े", "यकृत", "गुर्दे", "मस्तिष्क",
        "तंत्रिका", "प्रतिरक्षा", "एंटीबायोटिक", "सर्जरी",
        "चिकित्सक", "निदान", "जांच", "खुराक", "साइड इफेक्ट",
        "बुखार", "दर्द", "सूजन", "रक्तचाप", "मधुमेह",
        # English
        "medicine", "treatment", "symptoms", "disease", "infection",
        "vaccine", "blood", "heart", "lungs", "liver", "kidneys",
        "brain", "nervous", "immune", "antibiotic", "surgery",
        "doctor", "diagnosis", "test", "dose", "mg", "side effect",
        "fever", "pain", "swelling", "blood pressure", "diabetes",
        "inflammation", "chronic", "acute", "prescription", "therapy"
    ]

    scores = []
    for pred, ref in zip(predictions, references):
        pred_lower = pred.lower()
        ref_lower  = ref.lower()

        ref_kws  = [kw for kw in medical_keywords if kw.lower() in ref_lower]
        pred_kws = [kw for kw in medical_keywords if kw.lower() in pred_lower]

        if not ref_kws:
            scores.append(0.5)   # Neutral — reference has no medical keywords
            continue

        overlap = len(set(pred_kws) & set(ref_kws))
        scores.append(min(overlap / len(ref_kws), 1.0))

    return float(np.mean(scores))


# ── Per-model evaluation ───────────────────────────────────────────────────────
def evaluate_model(
    model_key: str,
    model_name: str,
    model_path: str,
    test_data: list,
    adapter_path: Path = None
) -> dict:
    """
    Evaluate one model variant. test_data is passed in — not reloaded.

    FIX: model_key field added so compare_models.py can look up by key
         instead of fragile positional indexing.
    """
    print(f"\n{'='*60}")
    print(f"🔬 Evaluating: {model_name}")
    print(f"{'='*60}")

    model, tokenizer = load_model(model_path, adapter_path)

    system_prompt = extract_system_prompt(test_data[0])

    print(f"\n📝 Generating answers for {len(test_data)} questions...")
    predictions, references = [], []

    for sample in tqdm(test_data, desc=model_name):
        question, reference = extract_qa(sample)
        pred = generate_answer(model, tokenizer, question, system_prompt)
        predictions.append(pred)
        references.append(reference if reference else "[EMPTY]")

    print("\n📊 Computing metrics...")
    rouge_l  = compute_rouge_l(predictions, references)
    bscore   = compute_bertscore(predictions, references)
    med_acc  = compute_medical_accuracy(predictions, references)

    print(f"\n📈 Results for {model_name}:")
    print(f"   ROUGE-L:          {rouge_l:.4f}")
    print(f"   BERTScore (multi):{bscore:.4f}")
    print(f"   Medical Accuracy: {med_acc:.4f}")

    # Free VRAM before loading the next model
    del model
    torch.cuda.empty_cache()

    return {
        "model_key":        model_key,           # FIX: keyed lookup for compare_models.py
        "model":            model_name,
        "num_samples":      len(test_data),
        "rouge_l":          round(rouge_l, 4),
        "bertscore":        round(bscore, 4),
        "medical_accuracy": round(med_acc, 4),
        "predictions":      predictions[:5],     # First 5 for inspection
        "references":       references[:5]
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("=" * 60)
    print("🏥 MedQA-Hindi Model Evaluation")
    print("=" * 60)
    print(f"   Models    : {args.models}")
    print(f"   Samples   : {args.max_samples}")
    print(f"   Device    : {DEVICE}")

    # Load test data once
    test_file = Path(args.test_file) if args.test_file else DATA_DIR / "test.jsonl"
    test_data = load_test_data(test_file, args.max_samples)

    # FIX: base model corrected to 0.5B — matches the local QLoRA training pipeline.
    # Comparing fine-tuned 0.5B vs base 1.5B is meaningless for showing improvement.
    model_registry = {
        "base": {
            "name":    "Base Model (Qwen2.5-0.5B)",
            "path":    "Qwen/Qwen2.5-0.5B-Instruct",
            "adapter": None
        },
        "qlora": {
            "name":    "QLoRA Fine-tuned (0.5B)",
            "path":    r"D:\B.TECH\Projects\medqa-hindi\training\outputs\qlora_rtx2050_full\final_merged",
            "adapter": None
        },
        "dpo": {
            "name":    "DPO Aligned (0.5B)",
            "path":    r"D:\B.TECH\Projects\medqa-hindi\training\outputs\dpo\final_merged",
            "adapter": None
        }
    }

    all_results = []

    for key in args.models:
        info = model_registry[key]
        model_path = info["path"]

        # Skip fine-tuned models that haven't been trained yet
        if key != "base" and not Path(model_path).exists():
            print(f"\n⚠️  Skipping {info['name']} — not found at {model_path}")
            continue

        results = evaluate_model(
            model_key=key,
            model_name=info["name"],
            model_path=model_path,
            test_data=test_data,
            adapter_path=Path(info["adapter"]) if info["adapter"] else None
        )

        if results:
            all_results.append(results)

    if not all_results:
        print("\n❌ No models were evaluated. Check your model paths.")
        return

    # Save results
    output_path = Path(args.output) if args.output else RESULTS_DIR / "evaluation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to {output_path}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("📊 Model Comparison")
    print("=" * 60)
    print(f"{'Model':<30} {'ROUGE-L':>10} {'BERTScore':>10} {'MedAcc':>10}")
    print("-" * 60)
    for r in all_results:
        print(
            f"{r['model']:<30} "
            f"{r['rouge_l']:>10.4f} "
            f"{r['bertscore']:>10.4f} "
            f"{r['medical_accuracy']:>10.4f}"
        )

    if len(all_results) > 1:
        base = next((r for r in all_results if r["model_key"] == "base"), None)
        if base:
            print("\n📈 Improvement over base:")
            for r in all_results:
                if r["model_key"] == "base":
                    continue
                rl_imp  = (r["rouge_l"]          - base["rouge_l"])          / max(base["rouge_l"], 1e-9)          * 100
                bs_imp  = (r["bertscore"]         - base["bertscore"])        / max(base["bertscore"], 1e-9)        * 100
                ma_imp  = (r["medical_accuracy"]  - base["medical_accuracy"]) / max(base["medical_accuracy"], 1e-9) * 100
                print(f"  {r['model']:<30}  ROUGE +{rl_imp:+.1f}%  BERTScore +{bs_imp:+.1f}%  MedAcc +{ma_imp:+.1f}%")

    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)
    print(f"\nNext step: python evaluation/compare_models.py")


if __name__ == "__main__":
    main()