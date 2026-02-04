"""Test chat template formatting for DPO data."""

import json
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    use_fast=True,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load sample preference data
with open("preference_pairs_cleaned.json", "r") as f:
    data = json.load(f)

sample = data["pairs"][0]


# Apply chat template formatting
def format_with_chat_template(example, tokenizer):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["input"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    chosen = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["chosen"]},
        ],
        tokenize=False,
    )
    rejected = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["rejected"]},
        ],
        tokenize=False,
    )
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


formatted = format_with_chat_template(sample, tokenizer)

print("=" * 80)
print("PROMPT:")
print("=" * 80)
print(formatted["prompt"])
print()

print("=" * 80)
print("CHOSEN:")
print("=" * 80)
print(formatted["chosen"])
print()

print("=" * 80)
print("REJECTED:")
print("=" * 80)
print(formatted["rejected"])
print()

# Print lengths
print("=" * 80)
print("LENGTHS:")
print(f"  Prompt tokens: {len(tokenizer.encode(formatted['prompt']))}")
print(f"  Chosen tokens: {len(tokenizer.encode(formatted['chosen']))}")
print(f"  Rejected tokens: {len(tokenizer.encode(formatted['rejected']))}")
