#!/usr/bin/env python3
"""
test_inference.py
Test your trained MedQA-Hindi model with 4-bit inference.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def main():
    print("=" * 60)
    print("🏥  MedQA-Hindi Inference Test")
    print("=" * 60)

    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path = "D:/B.TECH/Projects/medqa-hindi/training/outputs/qlora_rtx2050/final_adapter"

    print(f"\n📥 Loading base: {base_model_name}")

    # 4-bit inference (same as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"🔄 Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Test cases
    test_prompts = [
        {
            "lang": "Hindi",
            "prompt": "मरीज़ को बुखार और सिरदर्द है, क्या करना चाहिए?",
        },
        {
            "lang": "English",
            "prompt": "A patient has fever and headache. What should be done?",
        },
        {
            "lang": "Hindi",
            "prompt": "डायबिटीज़ के लक्षण क्या हैं?",
        },
        {
            "lang": "Hindi",
            "prompt": "उच्च रक्तचाप के सामान्य लक्षण और सावधानियाँ क्या हैं?",
        },
        {
            "lang": "English",
            "prompt": "What are the symptoms and precautions of hypertension?",
        },
    ]

    for test in test_prompts:
        print(f"\n{'='*60}")
        print(f"📝 [{test['lang']}] {test['prompt']}")
        print(f"{'='*60}")

        messages = [{"role": "user", "content": test["prompt"]}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.0,        # Deterministic
                do_sample=False,        # No randomness
                pad_token_id=tokenizer.pad_token_id,
            )

        # Extract only newly generated tokens (after prompt)
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        print(f"🤖 Response:\n{response}\n")


if __name__ == "__main__":
    main()