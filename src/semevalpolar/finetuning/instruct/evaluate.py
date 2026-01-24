import os.path

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, re

from semevalpolar.utils import get_project_root
from semevalpolar.finetuning.instruct.finetune import load_config

config = load_config()
adapter_path = os.path.join(get_project_root(), "predictions", "instruct", "final_model")

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

prompt = """Input:
The only sober person around this debate of illegal immigration is

Reasoning:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=2000,
    do_sample=False
)

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)
