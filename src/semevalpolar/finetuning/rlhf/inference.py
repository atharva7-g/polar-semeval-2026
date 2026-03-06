import asyncio
import json
import os
import re
from tqdm import tqdm
from openai import AsyncOpenAI

from semevalpolar.utils import get_project_root
from semevalpolar.finetuning.instruct.local_inference import (
    LocalResponse,
    LocalResponseUsage,
)


from tenacity import retry, stop_after_attempt, wait_exponential
import logging


MODEL_NAME = "google/gemma-3-27b-it"
MAX_NEW_TOKENS = 1024
LIMIT = None

MAX_CONCURRENT_REQUESTS = 30
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
async def generate_response(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7):
    async with semaphore:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=0.8,
            max_tokens=max_new_tokens,
            extra_body={"enable_thinking": False}
        )

    output_text = completion.choices[0].message.content
    finish_reason = completion.choices[0].finish_reason

    # If the output is None or empty, raise an error to trigger a retry
    if not output_text:
        logger.warning(f"Empty response received. Finish reason: {finish_reason}. Retrying...")
        raise ValueError(f"API returned empty content. Reason: {finish_reason}")

    usage = getattr(completion, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0)
    output_tokens = getattr(usage, "completion_tokens", 0)
    total_tokens = getattr(usage, "total_tokens", 0)

    cost = input_tokens * (0.04 / 1e6) + output_tokens * (0.15 / 1e6)

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
        root / "src" / "semevalpolar" / "finetuning" / "rlhf" / "prompt-v4.txt"
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

async def process_example(example, prompt_template, temperatures):
    """Processes a single dataset example across all temperatures concurrently."""
    input_text = example["input"]
    ground_truth = example.get("final answer (polarization)")
    prompt = prompt_template.format(input_text=input_text)

    example_results = {
        "input": input_text,
        "ground_truth": ground_truth,
        "completions": [],
    }

    # Create tasks for all temperatures for this specific example
    tasks = [
        generate_response(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=temp)
        for temp in temperatures
    ]
    
    # Run all temperature tasks concurrently
    responses = await asyncio.gather(*tasks)

    # Track usage for this specific example
    example_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0
    }

    for local_response in responses:
        usage = local_response.usage
        example_usage["input_tokens"] += usage.input_tokens
        example_usage["output_tokens"] += usage.output_tokens
        example_usage["total_tokens"] += usage.total_tokens
        example_usage["cost"] += usage.cost

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

    return example_results, example_usage

async def main():
    prompt_template = load_prompt_template()
    dataset = load_dataset()

    if LIMIT is not None:
        dataset = dataset[:LIMIT]
        print(f"Testing mode: Processing only first {LIMIT} examples")

    temperatures = [0.5, 0.7, 0.8, 0.9]

    # Create tasks for the entire dataset
    tasks = [
        process_example(example, prompt_template, temperatures)
        for example in dataset
    ]

    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_cost = 0.0

    # Use asyncio.as_completed to update tqdm as individual examples finish
    pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing examples")

    for completed_task in pbar:
        example_results, example_usage = await completed_task
        
        results.append(example_results)
        
        # Accumulate global totals
        total_input_tokens += example_usage["input_tokens"]
        total_output_tokens += example_usage["output_tokens"]
        total_tokens += example_usage["total_tokens"]
        total_cost += example_usage["cost"]

        pbar.set_postfix({
            "in": total_input_tokens,
            "out": total_output_tokens,
            "tok": total_tokens,
            "cost": f"${total_cost:.4f}",
        })

    root = get_project_root()
    output_path = (
        root / "src" / "semevalpolar" / "finetuning" / "rlhf" / "inference_results_v7.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    print(f"Total examples processed: {len(results)}")
    print(f"Total completions generated: {len(results) * len(temperatures)}")

if __name__ == "__main__":
    # Standard way to run an async main function
    asyncio.run(main())