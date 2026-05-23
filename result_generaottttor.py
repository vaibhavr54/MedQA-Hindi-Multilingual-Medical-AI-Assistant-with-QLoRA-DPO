from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

questions = [
    "मरीज़ को बुखार और सिरदर्द है, क्या करना चाहिए?",
    "डायबिटीज़ के लक्षण क्या हैं?",
    "उच्च रक्तचाप के लक्षण क्या हैं?",
    "What are symptoms of anemia?",
    "A patient has fever and headache. What should be done?"
]

predictions=[]

for q in questions:
    messages=[{"role":"user","content":q}]

    text=tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs=tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    output=model.generate(
        **inputs,
        max_new_tokens=100
    )

    response=tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    predictions.append(response)

print(json.dumps(predictions, indent=2, ensure_ascii=False))