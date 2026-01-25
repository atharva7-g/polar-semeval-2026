import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from semevalpolar.utils import get_project_root
from semevalpolar.finetuning.instruct.finetune import load_config

config = load_config()
adapter_path = os.path.join(
    get_project_root(), "predictions", "instruct", "final_model"
)

# Load tokenizer EXACTLY as used during training
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

prompt = """Input:
The only sober person around this debate of illegal immigration is

Reasoning:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id
)

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)
