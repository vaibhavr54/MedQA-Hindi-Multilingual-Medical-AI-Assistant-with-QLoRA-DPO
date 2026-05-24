from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float16)
print("Loading adapter...")
peft_model = PeftModel.from_pretrained(base, r"D:\B.TECH\Projects\medqa-hindi\training\outputs\qlora_rtx2050_full\final_adapter")
print("Merging...")
merged = peft_model.merge_and_unload()
print("Saving merged model...")
merged.save_pretrained(r"D:\B.TECH\Projects\medqa-hindi\training\outputs\qlora_rtx2050_full\final_merged_fp16")

tokenizer = AutoTokenizer.from_pretrained(r"D:\B.TECH\Projects\medqa-hindi\training\outputs\qlora_rtx2050_full\final_adapter")
tokenizer.save_pretrained(r"D:\B.TECH\Projects\medqa-hindi\training\outputs\qlora_rtx2050_full\final_merged_fp16")
print("Done!")

