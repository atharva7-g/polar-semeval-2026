import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from semevalpolar.utils import get_project_root
from semevalpolar.finetuning.instruct.finetune import load_config


def extract_label(text: str):
    match = re.search(r"Final Answer:\s*([01])", text)
    return int(match.group(1)) if match else None


def main():
    config = load_config()
    adapter_path = os.path.join(
        get_project_root(),
        "predictions",
        "instruct",
        "final_model"
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    base_model.resize_token_embeddings(len(tokenizer))

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
    print("=== MODEL OUTPUT ===")
    print(decoded)

    label = extract_label(decoded)
    print("\n=== EXTRACTED LABEL ===")
    print(label)


if __name__ == "__main__":
    main()
