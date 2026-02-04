#!/usr/bin/env python3
"""
Generate DPO preference pairs using a model.

Usage:
    # As a module:
    from generate_dpo_dataset import generate_dpo_pairs, save_dpo_pairs

    # Load data and generate pairs
    dataset = load_dataset()
    prompt_template = load_prompt_template()

    pairs = generate_dpo_pairs(
        dataset=dataset,
        prompt_template=prompt_template,
        model="meta-llama/Llama-3.1-8B-Instruct",
        limit=100,
        temperature=0.8
    )

    save_dpo_pairs(pairs, "output.json")

    # As a CLI script:
    python generate_dpo_dataset.py --limit 100 --temperature 0.8
"""

import json
import argparse
import ollama
from tqdm import tqdm
from semevalpolar.utils import get_project_root
from semevalpolar.finetuning.instruct.local_inference import (
    LocalResponse,
    LocalResponseUsage,
)


PROMPT_TEMPLATE_PATH = (
    "src/semevalpolar/finetuning/rlhf/dual_prompt/dpo_prompt_template.txt"
)
DATASET_PATH = "src/semevalpolar/finetuning/rlhf/data/response_dict_val.json"
OUTPUT_DIR = "src/semevalpolar/finetuning/rlhf/dual_prompt"

DEFAULT_MODEL = "llama3.3:latest"
DEFAULT_LIMIT = None
DEFAULT_TEMPERATURE = 0.7
MAX_NEW_TOKENS = 512


def generate_response(
    prompt: str,
    model: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> LocalResponse:
    """
    Generate a response using Ollama.

    Args:
        prompt: Input prompt
        model: Model name (uses DEFAULT_MODEL if None)
        max_new_tokens: Maximum tokens to generate
        temperature: Generation temperature

    Returns:
        LocalResponse object
    """
    if model is None:
        model = DEFAULT_MODEL

    response = ollama.generate(
        model=model,
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


def load_prompt_template(path: str | None = None) -> str:
    """Load DPO prompt template from file."""
    if path is None:
        path = PROMPT_TEMPLATE_PATH
    with open(get_project_root() / path, "r") as f:
        return f.read()


def load_dataset(path: str | None = None) -> list:
    """Load dataset from JSON file."""
    if path is None:
        path = DATASET_PATH
    with open(get_project_root() / path, "r") as f:
        data = json.load(f)
    return data["dataset"]


def parse_dpo_response(response_text: str) -> tuple:
    """
    Parse the DPO response to extract chosen and rejected outputs.
    Returns (chosen, rejected) tuples of (explanation, label).
    """
    chosen = None
    rejected = None

    import re

    # Find both output sections using [CHOSEN] and [REJECTED] tags
    chosen_match = re.search(
        r"\[CHOSEN\].*?\n(.*?)(?=\[REJECTED\]|$)",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )

    rejected_match = re.search(
        r"\[REJECTED\].*?\n(.*?)(?=\[|$)",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )

    def extract_explanation_and_label(section_text: str) -> dict | None:
        """Extract explanation and label from a section."""
        if not section_text:
            return None

        explanation = None
        label = None

        # Find Explanation section - look for "Explanation:" followed by text
        exp_match = re.search(
            r"Explanation:\s*(.*?)(?=\n\s*Label:|$)",
            section_text,
            re.DOTALL | re.IGNORECASE,
        )
        if exp_match:
            explanation = exp_match.group(1).strip()

        # Find Label section - look for "Label:" followed by 0 or 1
        label_match = re.search(r"Label:\s*([01])", section_text, re.IGNORECASE)
        if label_match:
            label = label_match.group(1).strip()

        if explanation and label:
            return {"explanation": explanation, "label": label}
        return None

    # Extract from [CHOSEN] section
    if chosen_match:
        chosen = extract_explanation_and_label(chosen_match.group(1))

    # Extract from [REJECTED] section
    if rejected_match:
        rejected = extract_explanation_and_label(rejected_match.group(1))

    return chosen, rejected


def create_pair_entry(input_text: str, chosen: dict, rejected: dict) -> dict:
    """Create a preference pair entry matching existing JSON format."""

    def format_output(explanation: str, label: str) -> str:
        return f'Statement:\n"{input_text}"\n\nExplanation:\n{explanation}\n\nLabel:\n{label}\n'

    return {
        "input": input_text,
        "chosen": format_output(chosen["explanation"], chosen["label"]),
        "rejected": format_output(rejected["explanation"], rejected["label"]),
    }


def generate_dpo_pairs(
    dataset: list,
    prompt_template: str,
    model: str | None = None,
    limit: int | None = None,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> list:
    """
    Generate DPO preference pairs for the dataset.

    Args:
        dataset: List of examples with 'input' and 'final answer (polarization)' fields
        prompt_template: DPO prompt template string
        model: Model to use for generation (None for default)
        limit: Number of examples to process (None for full dataset)
        temperature: Generation temperature
        max_new_tokens: Maximum new tokens to generate

    Returns:
        List of preference pair dictionaries
    """
    if limit:
        dataset = dataset[:limit]

    pairs = []
    errors = 0

    for example in tqdm(dataset, desc="Generating DPO pairs"):
        input_text = example["input"]
        ground_truth = example.get("final answer (polarization)", "1")

        # Format prompt with input text
        formatted_prompt = prompt_template.replace("{input_text}", input_text)

        try:
            response = generate_response(
                formatted_prompt,
                model=model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            response_text = response.output_text

            # Parse the response to extract chosen and rejected
            chosen, rejected = parse_dpo_response(response_text)

            # Debug output
            if not chosen or not rejected:
                tqdm.write(f"DEBUG: Failed to parse response:")
                tqdm.write(f"  Response text: {response_text[:200]}...")
                tqdm.write(f"  Chosen: {chosen}")
                tqdm.write(f"  Rejected: {rejected}")

            if chosen and rejected:
                # Ensure chosen has the correct (ground truth) label
                if chosen["label"] != ground_truth:
                    # Swap if chosen has wrong label
                    chosen, rejected = rejected, chosen

                # Create the pair entry
                pair = create_pair_entry(input_text, chosen, rejected)
                pairs.append(pair)
            else:
                errors += 1
                tqdm.write(f"Failed to parse response for: {input_text[:50]}...")

        except Exception as e:
            errors += 1
            tqdm.write(f"Error processing '{input_text[:50]}...': {e}")

    print(f"\nGenerated {len(pairs)} pairs ({errors} errors)")
    return pairs


def save_dpo_pairs(
    pairs: list,
    output_filename: str,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Save DPO pairs to a JSON file.

    Args:
        pairs: List of preference pair dictionaries
        output_filename: Name of output file
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    output_path = get_project_root() / output_dir / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({"pairs": pairs}, f, indent=2)

    print(f"DPO preference pairs saved to: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="Model to use for generation"
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help="Number of examples to process"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Generation temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dpo_preference_pairs.json",
        help="Output filename",
    )
    args = parser.parse_args()

    print("Loading prompts and dataset...")
    prompt_template = load_prompt_template()
    dataset = load_dataset()

    print(f"Dataset size: {len(dataset)}")
    if args.limit:
        print(f"Processing limit: {args.limit}")

    print("\nGenerating DPO preference pairs...")
    pairs = generate_dpo_pairs(
        dataset=dataset,
        prompt_template=prompt_template,
        model=args.model,
        limit=args.limit,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )

    # Save results
    save_dpo_pairs(pairs, args.output)


if __name__ == "__main__":
    main()
