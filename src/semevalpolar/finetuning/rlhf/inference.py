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


MODEL_NAME = "openai/gpt-oss-120b:nitro"
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
    reraise=True,
)
async def generate_response(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7):
    async with semaphore:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=0.8,
            max_tokens=max_new_tokens,
            extra_body={"enable_thinking": False},
        )

    output_text = completion.choices[0].message.content
    finish_reason = completion.choices[0].finish_reason

    # If the output is None or empty, raise an error to trigger a retry
    if not output_text:
        logger.warning(
            f"Empty response received. Finish reason: {finish_reason}. Retrying..."
        )
        raise ValueError(f"API returned empty content. Reason: {finish_reason}")

    usage = getattr(completion, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0)
    output_tokens = getattr(usage, "completion_tokens", 0)
    total_tokens = getattr(usage, "total_tokens", 0)

    cost = input_tokens * (0.039 / 1e6) + output_tokens * (0.19 / 1e6)

    return LocalResponse(
        output_text=output_text,
        usage=LocalResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
        ),
    )


def load_prompt_templates():
    """Load both prompt templates for dual-prompt inference.

    Prompt A (pro_polarization): Defends why text IS polarized
    Prompt B (anti_polarization): Defends why text is NOT polarized
    """
    root = get_project_root()

    with open(
        root
        / "src"
        / "semevalpolar"
        / "finetuning"
        / "rlhf"
        / "prompts"
        / "prompt-polarized.txt",
        "r",
    ) as f:
        prompt_a = f.read()

    # Prompt B: Anti-polarization (defend why it is NOT polarized)
    with open(
        root
        / "src"
        / "semevalpolar"
        / "finetuning"
        / "rlhf"
        / "prompts"
        / "prompt-notpolarized.txt",
        "r",
    ) as f:
        prompt_b = f.read()

    return {
        "A": prompt_a,  # pro_polarization
        "B": prompt_b,  # anti_polarization
    }


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


async def process_example(example, prompts, temperatures):
    """Processes a single dataset example using 2 prompts x 3 temperatures = 6 completions.

    Args:
        example: Dataset example with 'input' and ground truth
        prompts: Dict with keys 'A' and 'B' containing prompt templates
        temperatures: List of temperatures to use

    Returns:
        Tuple of (example_results, example_usage)
    """
    input_text = example["input"]
    ground_truth = example.get("final answer (polarization)")

    example_results = {
        "input": input_text,
        "ground_truth": ground_truth,
        "completions": [],
    }

    # Helper function to track which prompt_id and temperature generated each response
    async def run_task(prompt_text, prompt_id, temp):
        try:
            local_response = await generate_response(
                prompt_text, max_new_tokens=MAX_NEW_TOKENS, temperature=temp
            )
            return local_response, prompt_id, temp
        except Exception as e:
            logger.warning(f"Failed task {prompt_id} at temp {temp}: {e}")
            return None, prompt_id, temp

    # Build tasks: 2 prompts x 3 temperatures = 6 total
    tasks = []
    for temp in temperatures:
        # Prompt A (pro-polarization)
        prompt_a_text = prompts["A"].format(input_text=input_text)
        tasks.append(run_task(prompt_a_text, "A", temp))

        # Prompt B (anti-polarization)
        prompt_b_text = prompts["B"].format(input_text=input_text)
        tasks.append(run_task(prompt_b_text, "B", temp))

    # Run all 6 API calls concurrently
    responses = await asyncio.gather(*tasks)

    # Track usage for this specific example
    example_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
    }

    for local_response, prompt_id, temp in responses:
        # Skip failed generations (Option A: keep 5 that succeeded)
        if local_response is None:
            continue

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
        label_match = re.search(r"Final Answer:\s*([01])", local_response.output_text)
        final_label = label_match.group(1) if label_match else None

        example_results["completions"].append(
            {
                "prompt_id": prompt_id,  # Track which prompt (A or B)
                "temperature": temp,  # Track temperature
                "response": full_response,
                "final_label": final_label,
                "raw_response": local_response.output_text,
            }
        )

    return example_results, example_usage


async def main():
    prompts = load_prompt_templates()
    dataset = load_dataset()

    if LIMIT is not None:
        dataset = dataset[:LIMIT]
        print(f"Testing mode: Processing only first {LIMIT} examples")

    # 2 prompts x 3 temperatures = 6 completions per example
    temperatures = [0.6, 0.9, 1.2]
    num_completions_per_example = len(temperatures) * 2  # 2 prompts

    # Create tasks for the entire dataset
    tasks = [process_example(example, prompts, temperatures) for example in dataset]

    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_cost = 0.0

    # Use asyncio.as_completed to update tqdm as individual examples finish
    pbar = tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Processing examples"
    )

    for completed_task in pbar:
        example_results, example_usage = await completed_task

        results.append(example_results)

        # Accumulate global totals
        total_input_tokens += example_usage["input_tokens"]
        total_output_tokens += example_usage["output_tokens"]
        total_tokens += example_usage["total_tokens"]
        total_cost += example_usage["cost"]

        pbar.set_postfix(
            {
                "in": total_input_tokens,
                "out": total_output_tokens,
                "tok": total_tokens,
                "cost": f"${total_cost:.4f}",
            }
        )

    root = get_project_root()
    output_path = (
        root
        / "src"
        / "semevalpolar"
        / "finetuning"
        / "rlhf"
        / "inference_results_v8.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    print(f"Total examples processed: {len(results)}")
    print(f"Total completions generated: {len(results) * num_completions_per_example}")
    print(f"Prompts: A (pro-polarization), B (anti-polarization)")
    print(f"Temperatures: {temperatures}")


if __name__ == "__main__":
    # Standard way to run an async main function
    asyncio.run(main())
