import json
import re
from tqdm import tqdm
from semevalpolar.utils import get_project_root
from semevalpolar.finetuning.instruct.local_inference import (
    LocalResponse,
    LocalResponseUsage,
)
import os
from openai import OpenAI

MODEL_NAME = "qwen/qwen3-32b"
MAX_NEW_TOKENS = 256
LIMIT = 1

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def generate_response(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7):
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        top_p=0.8,
        max_tokens=max_new_tokens,
        extra_body={
            "enable_thinking": False
        }
    )

    output_text = completion.choices[0].message.content

    usage = getattr(completion, "usage", None)

    input_tokens = getattr(usage, "prompt_tokens", 0)
    output_tokens = getattr(usage, "completion_tokens", 0)
    total_tokens = getattr(usage, "total_tokens", 0)

    cost = input_tokens * (0.08 / 1e6) + output_tokens * (0.24 / 1e6)

    return LocalResponse(
        output_text=output_text,
        usage=LocalResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
        ),
    )

def load_prompt_template():
    root = get_project_root()
    prompt_path = (
        root / "src" / "semevalpolar" / "finetuning" / "rlhf" / "prompt-v5.txt"
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
        / "response_dict.json"
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

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_cost = 0.0

    pbar = tqdm(dataset, desc="Processing examples")

    for example in pbar:
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

            usage = local_response.usage
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
            total_tokens += usage.total_tokens
            total_cost += usage.cost



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

        pbar.set_postfix({
            "in": total_input_tokens,
            "out": total_output_tokens,
            "tok": total_tokens,
            "cost": f"${total_cost:.4f}",
        })
        results.append(example_results)

    root = get_project_root()
    output_path = (
        root / "src" / "semevalpolar" / "finetuning" / "rlhf" / "inference_results_v6_vllm.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Total examples processed: {len(results)}")
    print(f"Total completions generated: {len(results) * len(temperatures)}")


if __name__ == "__main__":
    main()
