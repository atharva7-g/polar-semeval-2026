import json
import re
import ollama
from pathlib import Path
from tqdm import tqdm
from semevalpolar.utils import get_project_root
from semevalpolar.finetuning.instruct.local_inference import (
    LocalResponse,
    LocalResponseUsage,
)

MODEL_NAME = "gemma3:27b"
MAX_NEW_TOKENS = 256
LIMIT = None


def generate_response(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7):
    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={
            "temperature": temperature,
            "top_p": 0.95,
            "num_predict": max_new_tokens,
        },
    )

    usage = response.get("usage", {})

    return LocalResponse(
        output_text=response["response"],
        usage=LocalResponseUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            cost=0.0,
        ),
    )


def load_prompt_template():
    root = get_project_root()
    prompt_path = (
        root / "src" / "semevalpolar" / "finetuning" / "rlhf" / "prompt-v2.txt"
    )
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
    prompt_template = load_prompt_template()
    dataset = load_dataset()

    if LIMIT is not None:
        dataset = dataset[:LIMIT]
        print(f"Testing mode: Processing only first {LIMIT} examples")

    temperatures = [0.5, 0.7, 0.8, 0.9]

    results = []

    for example in tqdm(dataset, desc="Processing examples"):
        input_text = example["input"]
        ground_truth = example.get("final answer (polarization)")
        prompt = prompt_template.format(input_text=input_text)

        example_results = {
            "input": input_text,
            "ground_truth": ground_truth,
            "completions": [],
        }

        for temp in temperatures:
            local_response = generate_response(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=temp,
            )

            # Extract the full reasoning + label block
            match = re.search(
                r"Reasoning:\s*[\s\S]*?Final Answer:\s*[01]",
                local_response.output_text,
            )
            full_response = match.group(0) if match else None

            # Extract just the final label (0 or 1)
            label_match = re.search(
                r"Final Answer:\s*([01])", local_response.output_text
            )
            final_label = label_match.group(1) if label_match else None

            example_results["completions"].append(
                {
                    "response": full_response,
                    "final_label": final_label,
                    "raw_response": local_response.output_text,
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
    print(f"Total completions generated: {len(results) * len(temperatures)}")


if __name__ == "__main__":
    main()
