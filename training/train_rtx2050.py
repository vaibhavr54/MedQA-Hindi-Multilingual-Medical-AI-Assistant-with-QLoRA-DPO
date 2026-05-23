#!/usr/bin/env python3
"""
MedQA-Hindi QLoRA — RTX 2050 Local Training v1.0 Stable
HF-native checkpointing with correct output_dir redirection.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.6"

import yaml
import torch
import math
import argparse
import json
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


CONFIG_PATH = Path("configs/qlora_rtx2050.yaml")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
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
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
    )


def setup_lora(config):
    l = config["qlora"]
    return LoraConfig(
        r=l["r"],
        lora_alpha=l["lora_alpha"],
        lora_dropout=l["lora_dropout"],
        target_modules=l["target_modules"],
        bias=l["bias"],
        task_type=l["task_type"],
    )


def load_model_and_tokenizer(config):
    model_name = config["model"]["base_model"]
    print(f"\n📥 Loading {model_name}")

    bnb_config = setup_quantization(config)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.use_cache = False

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ VRAM: {allocated:.2f}GB / {total:.2f}GB")

    return model, tokenizer


def prepare_model(model, config):
    print("\n🔧 Preparing model")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, setup_lora(config))
    model.print_trainable_parameters()
    return model


def load_datasets(config, tokenizer):
    d = config["data"]
    print("\n📊 Loading datasets")

    def load_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    train = load_jsonl(d["train_file"])
    val = load_jsonl(d["validation_file"])

    max_samples = d.get("max_samples")
    if max_samples:
        train = train[:max_samples]
        val = val[:max_samples // 5]

    def filter_long(data, tok, max_len):
        filtered = []
        for item in data:
            text = tok.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=False
            )
            if len(tok(text, add_special_tokens=False)["input_ids"]) <= max_len:
                filtered.append(item)
        return filtered

    max_len = config["training"]["max_seq_length"]
    print(f"   Filtering > {max_len} tokens...")
    train = filter_long(train, tokenizer, max_len)
    val = filter_long(val, tokenizer, max_len)

    train_dataset = Dataset.from_list([{"messages": x["messages"]} for x in train])
    val_dataset = Dataset.from_list([{"messages": x["messages"]} for x in val])

    print(f"✅ Train: {len(train_dataset)}")
    print(f"✅ Val:   {len(val_dataset)}")
    return train_dataset, val_dataset


def compute_total_steps(config, train_size):
    t = config["training"]
    batch_size = t["per_device_train_batch_size"]
    grad_accum = t["gradient_accumulation_steps"]
    epochs = t["num_train_epochs"]
    num_gpus = max(1, torch.cuda.device_count())

    effective_batch = batch_size * grad_accum * num_gpus
    steps_per_epoch = math.ceil(train_size / effective_batch)
    return steps_per_epoch * epochs


class ForceSaveCallback(TrainerCallback):
    """
    HF-native save with output_dir redirected to checkpoint folder.
    All state (adapters, optimizer, scheduler, RNG, trainer_state) lands in one place.
    """

    def __init__(self, output_dir, save_every=250):
        self.output_dir = Path(output_dir)
        self.save_every = save_every
        self.last_saved = 0
        self.trainer = None
        self.tokenizer = None

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step

        if step > 0 and step % self.save_every == 0 and step != self.last_saved:
            self.last_saved = step
            self._force_save(step)

        return control

    def _force_save(self, step):
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"💾 FORCE SAVE: checkpoint-{step}")

        if self.trainer is None:
            print("⚠️  Trainer not set — cannot save")
            print(f"{'='*60}\n")
            return

        # Redirect HF output to checkpoint directory
        old_dir = self.trainer.args.output_dir
        self.trainer.args.output_dir = str(checkpoint_dir)

        try:
            self.trainer.save_model()
            self.trainer.save_state()

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(checkpoint_dir)

            torch.cuda.empty_cache()

            # Verify
            weight_file = checkpoint_dir / "adapter_model.safetensors"
            state_file = checkpoint_dir / "trainer_state.json"

            if weight_file.exists() and state_file.exists():
                size_mb = weight_file.stat().st_size / 1e6
                print(f"✅ Saved: {size_mb:.1f} MB | Full HF-native checkpoint")
            else:
                print(f"⚠️  Save incomplete")

        finally:
            # Always restore original output_dir
            self.trainer.args.output_dir = old_dir

        print(f"{'='*60}\n")


def main():
    args = parse_args()

    print("=" * 60)
    print("🏥  MedQA-Hindi QLoRA — RTX 2050 v1.0 Stable")
    print("=" * 60)

    config = load_config()
    model, tokenizer = load_model_and_tokenizer(config)
    model = prepare_model(model, config)
    train_dataset, val_dataset = load_datasets(config, tokenizer)

    train_cfg = config["training"]

    total_steps = compute_total_steps(config, len(train_dataset))
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.05))
    print(f"\n📐 Total steps: {total_steps} | Warmup: {warmup_steps}")

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        optim=train_cfg["optim"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy="no",
        eval_strategy="no",
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        report_to=train_cfg["report_to"],
        remove_unused_columns=False,
        warmup_steps=warmup_steps,
    )

    print(f"\n💾 Full save every {train_cfg['save_steps']} steps")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
)

    # Setup callback with references
    save_callback = ForceSaveCallback(
        output_dir=train_cfg["output_dir"],
        save_every=train_cfg["save_steps"]
    )
    save_callback.trainer = trainer
    save_callback.tokenizer = tokenizer
    trainer.add_callback(save_callback)

    # Resume
    if args.resume_from_checkpoint:
        checkpoint = Path(args.resume_from_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        print(f"\n🔄 Resuming from: {checkpoint}")
        trainer.train(resume_from_checkpoint=str(checkpoint))
    else:
        print("\n🚀 Starting training...")
        trainer.train()

    # Final save
    print("\n💾 Final save...")
    final_path = Path(train_cfg["output_dir"]) / "final_adapter"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✅ Final: {final_path}")

    # Merge
    print("\n🔀 Merging...")
    try:
        merged = model.merge_and_unload()
        merged_path = Path(train_cfg["output_dir"]) / "final_merged"
        merged.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"✅ Merged: {merged_path}")
    except RuntimeError as e:
        print(f"\n⚠️  Merge failed: {e}")
        print("    Adapter saved — merge manually later")

    print("\n" + "=" * 60)
    print("✅  Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()