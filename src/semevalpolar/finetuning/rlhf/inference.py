import asyncio
import json
import re
import os
from pathlib import Path
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

from semevalpolar.utils import get_project_root
from semevalpolar.finetuning.instruct.local_inference import (
    LocalResponse,
    LocalResponseUsage,
)

MODEL_NAME = "qwen/qwen3-32b"
MAX_NEW_TOKENS = 32
TEMPERATURES = [0.5, 0.7, 0.8, 0.9]
CONCURRENCY = 20
LIMIT = None

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

semaphore = asyncio.Semaphore(CONCURRENCY)

RE_REASONING = re.compile(r"Reasoning:\s*[\s\S]*?Final Answer:\s*[01]")
RE_LABEL = re.compile(r"Final Answer:\s*([01])")


def load_prompt_template():
    root = get_project_root()
    path = root / "src/semevalpolar/finetuning/rlhf/prompt-v4.txt"
    return path.read_text()


def load_dataset():
    root = get_project_root()
    path = root / "src/semevalpolar/finetuning/rlhf/data/response_dict.json"
    data = json.loads(path.read_text())
    return data["dataset"]


async def generate_response(prompt, temperature):

    async with semaphore:
        response = await client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            max_output_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            top_p=0.95,
        )

    text = response.output_text
    usage = getattr(response, "usage", None)

    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    total_tokens = getattr(usage, "total_tokens", 0)

    cost = input_tokens * (0.08 / 1e6) + output_tokens * (0.24 / 1e6)

    reasoning_match = RE_REASONING.search(text)
    label_match = RE_LABEL.search(text)

    return {
        "response": reasoning_match.group(0) if reasoning_match else None,
        "final_label": label_match.group(1) if label_match else None,
        "raw_response": text,
        "usage": LocalResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
        ),
    }


async def process_example(example, prompt_template):

    input_text = example["input"]
    ground_truth = example.get("final answer (polarization)")

    prompt = prompt_template.format(input_text=input_text)

    tasks = [
        generate_response(prompt, temp)
        for temp in TEMPERATURES
    ]

    completions = await asyncio.gather(*tasks)

    return {
        "input": input_text,
        "ground_truth": ground_truth,
        "completions": completions,
    }


async def main():

    prompt_template = load_prompt_template()
    dataset = load_dataset()

    if LIMIT:
        dataset = dataset[:LIMIT]

    tasks = [
        process_example(example, prompt_template)
        for example in dataset
    ]

    results = []
    total_cost = 0

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):

        result = await coro
        results.append(result)

        for c in result["completions"]:
            total_cost += c["usage"].cost

    root = get_project_root()
    output_path = root / "src/semevalpolar/finetuning/rlhf/inference_results_async.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to:", output_path)
    print("Examples:", len(results))
    print("Total completions:", len(results) * len(TEMPERATURES))
    print("Total cost: $", round(total_cost, 4))


if __name__ == "__main__":
    asyncio.run(main())