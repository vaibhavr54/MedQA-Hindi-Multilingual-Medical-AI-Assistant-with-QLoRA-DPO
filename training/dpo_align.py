#!/usr/bin/env python3
"""
MedQA-Hindi DPO Alignment Script
Aligns QLoRA-finetuned model using Direct Preference Optimization.

Fixes applied:
  1. training_model vs base_model separation:
       training_model → QLoRA final_merged (DPO trains from here)
       base_model     → Qwen/Qwen2.5-0.5B-Instruct (clean base for merge only)
  2. merge_and_unload() reloads on fp16 — avoids quantized merge crash
  3. processing_class= instead of tokenizer= (TRL 0.9+)
  4. DPOConfig instead of TrainingArguments (TRL 0.9+)
  5. precompute_ref_log_probs=True for correct implicit reference behaviour
  6. Dataset capped at 15K train / 2K val — 136K samples = 10+ hrs on RTX 2050
  7. argparse resume support
  8. gradient_checkpointing: model.config.use_cache=False set explicitly
"""

import yaml
import torch
import json
import argparse
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / "configs" / "dpo.yaml"

MAX_TRAIN_SAMPLES = 15000
MAX_VAL_SAMPLES   = 2000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume DPO training from"
    )
    return parser.parse_args()


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def setup_quantization(config):
    q = config["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"]
    )


def setup_lora(config):
    l = config["qlora"]
    return LoraConfig(
        r=l["r"],
        lora_alpha=l["lora_alpha"],
        lora_dropout=l["lora_dropout"],
        target_modules=l["target_modules"],
        bias=l["bias"],
        task_type=l["task_type"]
    )


def load_model_and_tokenizer(config):
    """
    Load the QLoRA fine-tuned model for DPO training.

    KEY: uses config["model"]["training_model"] — the QLoRA final_merged.
    NOT base_model (Qwen/0.5B) — that is only used during merge_and_save().

    Flow:
      training_model (QLoRA merged) → DPO trains on top → saves DPO adapter
      base_model     (clean Qwen)   → merge_and_save() uses this for clean merge
    """
    # FIX: training_model, not base_model
    model_path = config["model"]["training_model"]
    print(f"📥 Loading QLoRA fine-tuned model for DPO: {model_path}")

    bnb_config = setup_quantization(config)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16
    )

    # Required when gradient_checkpointing=True
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total     = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   ✅ Model loaded | VRAM: {allocated:.2f}GB / {total:.2f}GB")
    else:
        print("   ✅ Model loaded (CPU)")

    return model, tokenizer


def load_dpo_datasets(config):
    """
    Load DPO preference datasets and cap size for RTX 2050.
    136K samples on 4GB VRAM = 10+ hours. Cap at 15K/2K.
    """
    data_config = config["data"]
    print("📊 Loading DPO datasets...")

    def load_jsonl(path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    train_data = load_jsonl(data_config["train_file"])
    val_data   = load_jsonl(data_config["validation_file"])

    # FIX: cap dataset — full 136K would run 10+ hours on RTX 2050
    train_data = train_data[:MAX_TRAIN_SAMPLES]
    val_data   = val_data[:MAX_VAL_SAMPLES]
    print(f"   📌 Capped to {len(train_data)} train / {len(val_data)} val samples")

    def format_dpo(samples):
        formatted = []
        for s in samples:
            if not all(k in s for k in ("prompt", "chosen", "rejected")):
                continue
            formatted.append({
                "prompt":   s["prompt"],
                "chosen":   s["chosen"],
                "rejected": s["rejected"]
            })
        return formatted

    train_dataset = Dataset.from_list(format_dpo(train_data))
    val_dataset   = Dataset.from_list(format_dpo(val_data))

    print(f"   ✅ Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    return train_dataset, val_dataset


def merge_and_save(base_model_path, adapter_path, merged_path, tokenizer):
    """
    Merge DPO adapter into a clean fp16 base model.

    MUST use clean Qwen/0.5B base here — NOT the QLoRA merged model.
    merge_and_unload() crashes on 4-bit quantized models, so we reload
    the base in fp16, attach the adapter, then merge cleanly.
    """
    print("\n🔀 Reloading clean base for merge...")
    print(f"   Base: {base_model_path}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"   🔌 Attaching DPO adapter from {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))

    print("   🔀 Merging...")
    merged_model = peft_model.merge_and_unload()

    merged_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"   ✅ Merged model saved to {merged_path}")

    del merged_model, peft_model, base_model
    torch.cuda.empty_cache()


def main():
    args = parse_args()

    print("=" * 60)
    print("🏥 MedQA-Hindi DPO Alignment")
    print("=" * 60)

    config       = load_config()
    train_config = config["training"]

    model, tokenizer   = load_model_and_tokenizer(config)
    train_dataset, val_dataset = load_dpo_datasets(config)
    peft_config        = setup_lora(config)

    # FIX: DPOConfig instead of TrainingArguments — beta and DPO params live here in TRL 0.9+
    dpo_args = DPOConfig(
        output_dir=train_config["output_dir"],
        num_train_epochs=train_config["num_train_epochs"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        warmup_ratio=train_config["warmup_ratio"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        optim=train_config["optim"],
        weight_decay=train_config["weight_decay"],
        max_grad_norm=train_config["max_grad_norm"],
        logging_steps=train_config["logging_steps"],
        save_strategy=train_config["save_strategy"],
        eval_strategy=train_config["eval_strategy"],
        load_best_model_at_end=train_config["load_best_model_at_end"],
        fp16=train_config["fp16"],
        bf16=train_config["bf16"],
        gradient_checkpointing=train_config["gradient_checkpointing"],
        report_to=train_config["report_to"],
        remove_unused_columns=False,
        # DPO-specific params
        beta=train_config["beta"],
        max_length=train_config["max_seq_length"],
        max_prompt_length=train_config["max_prompt_length"],
        max_completion_length=train_config["max_completion_length"],
        loss_type=train_config["loss_type"],
        precompute_ref_log_probs=True,  # required for correct implicit ref with quantized model
    )

    print("\n🚀 Initializing DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train (with optional resume)
    if args.resume_from_checkpoint:
        checkpoint = Path(args.resume_from_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        print(f"\n🔄 Resuming from: {checkpoint}")
        trainer.train(resume_from_checkpoint=str(checkpoint))
    else:
        print("\n🚀 Starting DPO training...")
        trainer.train()

    # Save DPO adapter
    adapter_path = Path(train_config["output_dir"]) / "final_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n💾 DPO adapter saved to {adapter_path}")

    # Merge: clean Qwen base + DPO adapter → final_merged
    merged_path = Path(train_config["output_dir"]) / "final_merged"
    try:
        merge_and_save(
            base_model_path=config["model"]["base_model"],  # clean Qwen/0.5B
            adapter_path=adapter_path,
            merged_path=merged_path,
            tokenizer=tokenizer
        )
    except Exception as e:
        print(f"\n⚠️  Merge failed: {e}")
        print("    Adapter is saved — merge manually later:")
        print(f"    base  = AutoModelForCausalLM.from_pretrained('{config['model']['base_model']}', torch_dtype=torch.float16)")
        print(f"    model = PeftModel.from_pretrained(base, '{adapter_path}')")
        print(f"    model.merge_and_unload().save_pretrained('{merged_path}')")

    print("\n" + "=" * 60)
    print("✅ DPO alignment complete!")
    print("=" * 60)
    print(f"""
Output files:
  DPO adapter : {adapter_path}
  Merged model: {merged_path}

Next step:
  python evaluation/evaluate.py
    """)


if __name__ == "__main__":
    main()