import json
import re
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from semevalpolar.utils import get_project_root

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 80
LIMIT = None


def load_model(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    return tokenizer, model


def generate_response(
    prompt, tokenizer, model, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7
):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    decoded = tokenizer.decode(generated, skip_special_tokens=True)

    match = re.search(r"Reasoning:\s*[\s\S]*?\n\nFinal label:\s*[01]", decoded)

    if not match:
        return None

    return match.group(0)


def load_prompt_template():
    root = get_project_root()
    prompt_path = root / "src" / "semevalpolar" / "finetuning" / "rlhf" / "simple-prompt.txt"
    with open(prompt_path, "r") as f:
        return f.read()


def load_dataset():
    root = get_project_root()
    dataset_path = (
        root
        / "src"
        / "semevalpolar"
        / "finetuning"
        / "rlhf"
        / "data"
        / "response_dict_val.json"
    )
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return data["dataset"]


def main():
    tokenizer, model = load_model()
    prompt_template = load_prompt_template()
    dataset = load_dataset()

    if LIMIT is not None:
        dataset = dataset[:LIMIT]
        print(f"Testing mode: Processing only first {LIMIT} examples")

    configs = [
        {"temperature": 0.5, "name": "Conservative"},
        {"temperature": 0.7, "name": "Default"},
        {"temperature": 0.8, "name": "Balanced"},
        {"temperature": 0.9, "name": "Creative"},
    ]

    results = []

    for example in tqdm(dataset, desc="Processing examples"):
        input_text = example["input"]
        prompt = prompt_template.format(input_text=input_text)

        example_results = {"input": input_text, "completions": []}

        for config in configs:
            response = generate_response(
                prompt,
                tokenizer,
                model,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=config["temperature"],
            )

            example_results["completions"].append(
                {
                    "config": config["name"],
                    "temperature": config["temperature"],
                    "response": response,
                }
            )

        results.append(example_results)

    root = get_project_root()
    output_path = (
        root / "src" / "semevalpolar" / "finetuning" / "rlhf" / "inference_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Total examples processed: {len(results)}")
    print(f"Total completions generated: {len(results) * len(configs)}")


if __name__ == "__main__":
    main()
