# #!/usr/bin/env python3
# """
# MedQA-Hindi QLoRA Fine-Tuning Script
# Updated: Auto-backup to Drive every checkpoint, step-based saving, speed optimized.
# """

# import os
# import yaml
# import torch
# import json
# import shutil
# from pathlib import Path
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     TrainingArguments,
#     BitsAndBytesConfig,
#     TrainerCallback,
# )
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from trl import SFTTrainer
# from datasets import Dataset

# CONFIG_PATH = Path(__file__).parent / "configs" / "qlora.yaml"

# def load_config():
#     with open(CONFIG_PATH, "r") as f:
#         return yaml.safe_load(f)

# def setup_quantization(config):
#     q_config = config["quantization"]
#     return BitsAndBytesConfig(
#         load_in_4bit=q_config["load_in_4bit"],
#         bnb_4bit_compute_dtype=getattr(torch, q_config["bnb_4bit_compute_dtype"]),
#         bnb_4bit_use_double_quant=q_config["bnb_4bit_use_double_quant"],
#         bnb_4bit_quant_type=q_config["bnb_4bit_quant_type"]
#     )

# def setup_lora(config):
#     lora_config = config["qlora"]
#     return LoraConfig(
#         r=lora_config["r"],
#         lora_alpha=lora_config["lora_alpha"],
#         lora_dropout=lora_config["lora_dropout"],
#         target_modules=lora_config["target_modules"],
#         bias=lora_config["bias"],
#         task_type=lora_config["task_type"]
#     )

# def load_model_and_tokenizer(config):
#     model_name = config["model"]["base_model"]
#     print(f"📥 Loading base model: {model_name}")
#     bnb_config = setup_quantization(config)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=bnb_config,
#         device_map="auto",
#         trust_remote_code=True,
#         torch_dtype=torch.float16
#     )
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name,
#         trust_remote_code=True,
#         padding_side="right"
#     )
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.pad_token_id = tokenizer.eos_token_id
#     print(f"   ✅ Model loaded (VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB)")
#     return model, tokenizer

# def prepare_model(model, config):
#     print("🔧 Preparing model for training...")
#     model = prepare_model_for_kbit_training(model)
#     lora_config = setup_lora(config)
#     model = get_peft_model(model, lora_config)
#     model.print_trainable_parameters()
#     return model

# def load_datasets(config):
#     data_config = config["data"]
#     print("📊 Loading datasets...")
#     def load_jsonl(path):
#         samples = []
#         with open(path, "r", encoding="utf-8") as f:
#             for line in f:
#                 samples.append(json.loads(line))
#         return samples
#     train_data = load_jsonl(data_config["train_file"])
#     val_data = load_jsonl(data_config["validation_file"])
#     max_samples = data_config.get("max_samples")
#     if max_samples:
#         train_data = train_data[:max_samples]
#         val_data = val_data[:max_samples//5]
#     train_dataset = Dataset.from_list([{"messages": s["messages"]} for s in train_data])
#     val_dataset = Dataset.from_list([{"messages": s["messages"]} for s in val_data])
#     print(f"   ✅ Train: {len(train_dataset)}, Val: {len(val_dataset)}")
#     return train_dataset, val_dataset

# class DriveBackupCallback(TrainerCallback):
#     """Auto-backup every checkpoint to Google Drive."""
#     def on_save(self, args, state, control, **kwargs):
#         checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
#         drive_dir = Path("/content/drive/MyDrive/medqa-hindi/checkpoints") / f"checkpoint-{state.global_step}"
#         if checkpoint_dir.exists():
#             drive_dir.parent.mkdir(parents=True, exist_ok=True)
#             if drive_dir.exists():
#                 shutil.rmtree(drive_dir)
#             shutil.copytree(checkpoint_dir, drive_dir)
#             print(f"☁️  Drive backup: checkpoint-{state.global_step}")

# def main():
#     print("=" * 60)
#     print("🏥 MedQA-Hindi QLoRA Fine-Tuning")
#     print("=" * 60)
    
#     config = load_config()
#     model, tokenizer = load_model_and_tokenizer(config)
#     model = prepare_model(model, config)
#     train_dataset, val_dataset = load_datasets(config)
    
#     train_config = config["training"]
#     training_args = TrainingArguments(
#         output_dir=train_config["output_dir"],
#         num_train_epochs=train_config["num_train_epochs"],
#         per_device_train_batch_size=train_config["per_device_train_batch_size"],
#         per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
#         gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
#         learning_rate=train_config["learning_rate"],
#         lr_scheduler_type=train_config["lr_scheduler_type"],
#         optim=train_config["optim"],
#         weight_decay=train_config["weight_decay"],
#         max_grad_norm=train_config["max_grad_norm"],
#         logging_steps=train_config["logging_steps"],
#         save_strategy=train_config["save_strategy"],
#         save_steps=train_config.get("save_steps", 500),
#         save_total_limit=train_config.get("save_total_limit", 3),
#         eval_strategy=train_config["eval_strategy"],
#         eval_steps=train_config.get("eval_steps", 500),
#         load_best_model_at_end=train_config["load_best_model_at_end"],
#         metric_for_best_model=train_config["metric_for_best_model"],
#         fp16=train_config["fp16"],
#         bf16=train_config["bf16"],
#         gradient_checkpointing=train_config["gradient_checkpointing"],
#         report_to=train_config["report_to"],
#         remove_unused_columns=False,
#         warmup_ratio=train_config.get("warmup_ratio", 0.03),
#     )
    
#     # FORCE step-based checkpointing
#     training_args.save_strategy = "steps"
#     training_args.save_steps = 500
#     training_args.save_total_limit = 3
    
#     print(f"\n💾 Checkpoint config: every {training_args.save_steps} steps")
#     print(f"   Strategy: {training_args.save_strategy}")
#     print(f"   Keep last {training_args.save_total_limit} checkpoints")
#     print(f"   Auto-backup to Drive: ENABLED")
    
#     print("\n🚀 Starting training...")
#     try:
#         trainer = SFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=val_dataset,
#             processing_class=tokenizer,
#         )
#     except TypeError:
#         try:
#             trainer = SFTTrainer(
#                 model=model,
#                 args=training_args,
#                 train_dataset=train_dataset,
#                 eval_dataset=val_dataset,
#                 tokenizer=tokenizer,
#                 max_seq_length=train_config["max_seq_length"],
#                 dataset_text_field="messages",
#                 packing=False,
#             )
#         except TypeError:
#             trainer = SFTTrainer(
#                 model=model,
#                 args=training_args,
#                 train_dataset=train_dataset,
#                 eval_dataset=val_dataset,
#                 tokenizer=tokenizer,
#             )
    
#     # Add auto-backup callback
#     trainer.add_callback(DriveBackupCallback())
    
#     # Train
#     trainer.train()
    
#     # Save final
#     adapter_path = Path(train_config["output_dir"]) / "final_adapter"
#     adapter_path.mkdir(parents=True, exist_ok=True)
#     trainer.save_model(adapter_path)
#     print(f"\n💾 LoRA adapters saved to {adapter_path}")
#     tokenizer.save_pretrained(adapter_path)
    
#     # Backup final to Drive
#     drive_final = Path("/content/drive/MyDrive/medqa-hindi/outputs/qlora_final")
#     if drive_final.parent.exists():
#         if drive_final.exists():
#             shutil.rmtree(drive_final)
#         shutil.copytree(adapter_path, drive_final)
#         print(f"☁️  Drive backup: final_adapter")
    
#     # Merge and save
#     print("\n🔀 Merging adapters with base model...")
#     merged_path = Path(train_config["output_dir"]) / "final_merged"
#     merged_path.mkdir(parents=True, exist_ok=True)
#     merged_model = model.merge_and_unload()
#     merged_model.save_pretrained(merged_path)
#     tokenizer.save_pretrained(merged_path)
#     print(f"💾 Merged model saved to {merged_path}")
    
#     # Backup merged to Drive
#     drive_merged = Path("/content/drive/MyDrive/medqa-hindi/outputs/qlora_merged")
#     if drive_merged.parent.exists():
#         if drive_merged.exists():
#             shutil.rmtree(drive_merged)
#         shutil.copytree(merged_path, drive_merged)
#         print(f"☁️  Drive backup: final_merged")
    
#     print("\n" + "=" * 60)
#     print("✅ QLoRA fine-tuning complete!")
#     print("=" * 60)
#     print(f"""
# Output files:
#   - Local: {adapter_path}, {merged_path}
#   - Drive: /content/drive/MyDrive/medqa-hindi/checkpoints/
#            /content/drive/MyDrive/medqa-hindi/outputs/

# Next step: python training/dpo_align.py
#     """)

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
MedQA-Hindi QLoRA Fine-Tuning v1.0 Stable
Supports: fresh training, resume from checkpoint, auto Drive backup, T4/A100/H100/L4 auto-detect.
"""

import yaml
import torch
import json
import shutil
import argparse
import math
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


CONFIG_PATH = Path(__file__).parent / "configs" / "qlora.yaml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from"
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


def get_mixed_precision_settings(config):
    """Auto-detect bf16 support or fall back to config + fp16 default."""
    gpu_name = "CPU"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        bf16_supported = (
            "A100" in gpu_name
            or "L4" in gpu_name
            or "H100" in gpu_name
        )
    else:
        bf16_supported = False

    fp16 = config["training"].get("fp16", not bf16_supported)
    bf16 = config["training"].get("bf16", bf16_supported)

    if fp16 and bf16:
        print("⚠️  Both fp16 and bf16 enabled. Defaulting to fp16.")
        bf16 = False

    print(f"🎯 Precision: fp16={fp16}, bf16={bf16} ({gpu_name})")
    return fp16, bf16


def load_model_and_tokenizer(config):
    model_name = config["model"]["base_model"]
    print(f"\n📥 Loading {model_name}")

    bnb_config = setup_quantization(config)
    fp16, bf16 = get_mixed_precision_settings(config)

    dtype = torch.bfloat16 if bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=dtype,
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

    print(f"✅ VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


def prepare_model(model, config):
    print("\n🔧 Preparing model for training")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, setup_lora(config))
    model.print_trainable_parameters()
    return model


def load_datasets(config):
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

    train_dataset = Dataset.from_list([{"messages": x["messages"]} for x in train])
    val_dataset = Dataset.from_list([{"messages": x["messages"]} for x in val])

    print(f"✅ Train: {len(train_dataset)}")
    print(f"✅ Val:   {len(val_dataset)}")
    return train_dataset, val_dataset


def compute_total_steps(config, train_size):
    """Compute total training steps dynamically from config."""
    t = config["training"]
    batch_size = t["per_device_train_batch_size"]
    grad_accum = t["gradient_accumulation_steps"]
    epochs = t["num_train_epochs"]
    num_gpus = max(1, torch.cuda.device_count())

    effective_batch = batch_size * grad_accum * num_gpus
    steps_per_epoch = math.ceil(train_size / effective_batch)
    return steps_per_epoch * epochs


def backup_to_drive(local_path: Path, drive_path: Path):
    """Copy local directory to Google Drive if mounted."""
    if not Path("/content/drive").exists():
        return False

    drive_path.parent.mkdir(parents=True, exist_ok=True)
    if drive_path.exists():
        shutil.rmtree(drive_path)
    shutil.copytree(local_path, drive_path)
    return True


class DriveBackupCallback(TrainerCallback):
    """Auto-backup every checkpoint to Google Drive."""

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        drive_dir = Path("/content/drive/MyDrive/medqa-hindi/checkpoints") / f"checkpoint-{state.global_step}"

        if checkpoint_dir.exists() and backup_to_drive(checkpoint_dir, drive_dir):
            print(f"☁️  Drive backup: checkpoint-{state.global_step}")


def build_trainer(model, training_args, train_dataset, val_dataset, tokenizer, config):
    """Build SFTTrainer with fallbacks for different TRL versions."""
    train_cfg = config["training"]

    try:
        return SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
        )
    except TypeError:
        pass

    try:
        return SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            max_seq_length=train_cfg.get("max_seq_length", 2048),
            dataset_text_field="messages",
            packing=False,
        )
    except TypeError:
        pass

    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )


def validate_training_config(config):
    """Catch config inconsistencies before training starts."""
    t = config["training"]
    if t.get("load_best_model_at_end", False):
        save_steps = t.get("save_steps", 500)
        eval_steps = t.get("eval_steps", 500)
        if save_steps != eval_steps:
            raise ValueError(
                f"load_best_model_at_end=True requires save_steps ({save_steps}) "
                f"to equal eval_steps ({eval_steps})"
            )


def main():
    args = parse_args()

    print("=" * 60)
    print("🏥  MedQA-Hindi QLoRA Fine-Tuning v1.0 Stable")
    print("=" * 60)

    config = load_config()
    validate_training_config(config)

    model, tokenizer = load_model_and_tokenizer(config)
    model = prepare_model(model, config)
    train_dataset, val_dataset = load_datasets(config)

    train_cfg = config["training"]

    total_steps = compute_total_steps(config, len(train_dataset))
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.03))
    print(f"\n📐 Total steps: {total_steps} | Warmup: {warmup_steps}")

    fp16, bf16 = get_mixed_precision_settings(config)

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
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        evaluation_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 500),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", False),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=False,
        warmup_steps=warmup_steps,
    )

    print(f"\n💾 Checkpoint every {training_args.save_steps} steps")
    print(f"   Keep last {training_args.save_total_limit} checkpoints")

    trainer = build_trainer(model, training_args, train_dataset, val_dataset, tokenizer, config)
    trainer.add_callback(DriveBackupCallback())

    # Training with checkpoint validation
    print("\n🚀 Starting training...")
    if args.resume_from_checkpoint:
        checkpoint = Path(args.resume_from_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        print(f"\n🔄 Resuming from: {checkpoint}")
        trainer.train(resume_from_checkpoint=str(checkpoint))
    else:
        trainer.train()

    # Save final adapter
    print("\n💾 Saving final adapter...")
    adapter_path = Path(train_cfg["output_dir"]) / "final_adapter"
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    if backup_to_drive(adapter_path, Path("/content/drive/MyDrive/medqa-hindi/outputs/qlora_final")):
        print("☁️  Drive backup: final_adapter")

    # Merge with safe fallback
    print("\n🧹 Clearing memory before merge...")
    del trainer
    torch.cuda.empty_cache()

    print("\n🔀 Merging adapters with base model...")
    try:
        merged_model = model.merge_and_unload()
        merged_path = Path(train_cfg["output_dir"]) / "final_merged"
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)

        if backup_to_drive(merged_path, Path("/content/drive/MyDrive/medqa-hindi/outputs/qlora_merged")):
            print("☁️  Drive backup: final_merged")

    except RuntimeError as e:
        print(f"\n⚠️  Merge failed: {e}")
        print("    LoRA adapter was already saved. Run merge manually later:")
        print(f"    python -c \"from peft import AutoPeftModelForCausalLM; "
              f"m = AutoPeftModelForCausalLM.from_pretrained('{adapter_path}'); "
              f"m.merge_and_unload().save_pretrained('{train_cfg['output_dir']}/final_merged')\"")

    print("\n" + "=" * 60)
    print("✅  QLoRA fine-tuning complete!")
    print("=" * 60)
    print(f"""
Outputs:
  - Adapter:  {adapter_path}
  - Merged:   {Path(train_cfg['output_dir']) / 'final_merged'} (if merge succeeded)
  - Drive:    /content/drive/MyDrive/medqa-hindi/outputs/

Next: python training/dpo_align.py
""")


if __name__ == "__main__":
    main()