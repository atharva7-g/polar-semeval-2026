#!/usr/bin/env python3
"""
Dual-prompt inference for creating preference pairs.
Uses Prompt 1 (Ground Truth Explainer) and Prompt 2 (Judgment).
"""

import json
import re
from pathlib import Path
from tqdm import tqdm
from semevalpolar.utils import get_project_root
from semevalpolar.finetuning.rlhf.inference import (
    generate_response,
    MAX_NEW_TOKENS,
)

# Configuration
PROMPT1_PATH = "src/semevalpolar/finetuning/rlhf/prompts/ground_truth_explainer_v1.txt"
PROMPT2_PATH = "src/semevalpolar/finetuning/rlhf/prompts/judgment_prompt_v1.txt"
DATASET_PATH = "src/semevalpolar/finetuning/rlhf/data/response_dict_val.json"
OUTPUT_DIR = "src/semevalpolar/finetuning/rlhf/dual_prompt"
INTERMEDIATE_FILE = "intermediate_results.json"
PAIRS_FILE = "preference_pairs.json"

PROMPT2_RUNS = 4
PROMPT2_TEMPS = [0.5, 0.7, 0.8, 0.9]
LIMIT = None  # Set to a number for testing, e.g., 10


def load_prompt(path):
    """Load prompt template from file."""
    with open(get_project_root() / path, "r") as f:
        return f.read()


def extract_label(response_text):
    """Extract label (0 or 1) from response text."""
    match = re.search(r"Label:\s*([01])", response_text)
    return match.group(1) if match else None


def load_dataset():
    """Load dataset from JSON file."""
    with open(get_project_root() / DATASET_PATH, "r") as f:
        data = json.load(f)
    return data["dataset"]


def run_prompt1(input_text, ground_truth, prompt_template):
    """Run Prompt 1 (Ground Truth Explainer) once."""
    formatted_prompt = prompt_template.format(
        input_text=input_text, ground_truth_label=ground_truth
    )

    response = generate_response(
        formatted_prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0
    )

    label = extract_label(response.output_text)

    return {"response": response.output_text, "label": label, "temperature": 0.0}


def run_prompt2(input_text, prompt_template):
    """Run Prompt 2 (Judgment) multiple times with different temperatures."""
    formatted_prompt = prompt_template.format(input_text=input_text)

    runs = []
    for temp in PROMPT2_TEMPS[:PROMPT2_RUNS]:
        response = generate_response(
            formatted_prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=temp
        )

        label = extract_label(response.output_text)

        runs.append(
            {"response": response.output_text, "label": label, "temperature": temp}
        )

    return runs


def create_preference_pairs(example_result):
    """Create up to 2 preference pairs where Prompt 2 label differs from ground truth."""
    ground_truth = example_result["ground_truth"]
    prompt1_response = example_result["prompt1"]["response"]
    input_text = example_result["input"]

    pairs = []

    # Find Prompt 2 responses with wrong labels
    wrong_responses = []
    for run in example_result["prompt2_runs"]:
        if run["label"] != ground_truth:
            wrong_responses.append(run["response"])

    # Create up to 2 pairs
    for rejected_response in wrong_responses[:2]:
        pairs.append(
            {
                "input": input_text,
                "chosen": prompt1_response,
                "rejected": rejected_response,
            }
        )

    return pairs


def main():
    # Load prompts and dataset
    print("Loading prompts and dataset...")
    prompt1_template = load_prompt(PROMPT1_PATH)
    prompt2_template = load_prompt(PROMPT2_PATH)
    dataset = load_dataset()

    # Apply limit if set
    if LIMIT is not None:
        dataset = dataset[:LIMIT]
        print(f"Testing mode: Processing only first {LIMIT} examples")

    # Process all examples
    print(f"\nProcessing {len(dataset)} examples...")
    intermediate_results = []

    for example in tqdm(dataset, desc="Running inference"):
        input_text = example["input"]
        ground_truth = example.get("final answer (polarization)")

        # Run Prompt 1 (Ground Truth Explainer)
        prompt1_result = run_prompt1(input_text, ground_truth, prompt1_template)

        # Run Prompt 2 (Judgment) multiple times
        prompt2_results = run_prompt2(input_text, prompt2_template)

        # Store intermediate result
        intermediate_results.append(
            {
                "input": input_text,
                "ground_truth": ground_truth,
                "prompt1": prompt1_result,
                "prompt2_runs": prompt2_results,
            }
        )

    # Save intermediate results
    output_dir = get_project_root() / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    intermediate_output = {
        "config": {
            "limit": LIMIT,
            "prompt2_runs": PROMPT2_RUNS,
            "prompt2_temperatures": PROMPT2_TEMPS[:PROMPT2_RUNS],
        },
        "results": intermediate_results,
    }

    intermediate_path = output_dir / INTERMEDIATE_FILE
    with open(intermediate_path, "w") as f:
        json.dump(intermediate_output, f, indent=2)

    print(f"\nIntermediate results saved to: {intermediate_path}")

    # Create preference pairs
    print("\nCreating preference pairs...")
    all_pairs = []
    examples_with_pairs = 0

    for result in intermediate_results:
        pairs = create_preference_pairs(result)
        if pairs:
            examples_with_pairs += 1
            all_pairs.extend(pairs)

    # Calculate statistics
    stats = {
        "total_examples": len(intermediate_results),
        "examples_with_pairs": examples_with_pairs,
        "total_pairs_created": len(all_pairs),
        "max_pairs_per_example": 2,
    }

    # Save preference pairs
    pairs_output = {"stats": stats, "pairs": all_pairs}

    pairs_path = output_dir / PAIRS_FILE
    with open(pairs_path, "w") as f:
        json.dump(pairs_output, f, indent=2)

    print(f"\nPreference pairs saved to: {pairs_path}")
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total examples processed: {stats['total_examples']}")
    print(f"Examples with pairs: {stats['examples_with_pairs']}")
    print(f"Total pairs created: {stats['total_pairs_created']}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
