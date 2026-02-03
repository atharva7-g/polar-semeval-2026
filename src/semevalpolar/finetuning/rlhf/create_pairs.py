import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from semevalpolar.utils import get_project_root


def classify_outcome(final_label: Optional[str], ground_truth: str) -> str:
    """Classify the outcome based on final_label and ground_truth."""
    if final_label is None:
        return "INVALID"

    pred = final_label.strip()
    gt = ground_truth.strip()

    if pred == gt:
        return "CORRECT"
    elif pred == "1" and gt == "0":
        return "FP"  # False Positive
    elif pred == "0" and gt == "1":
        return "FN"  # False Negative
    else:
        return "INVALID"


def get_priority(outcome: str) -> int:
    """Return priority for outcome ordering."""
    priorities = {"CORRECT": 3, "FP": 2, "FN": 1, "INVALID": 0}
    return priorities.get(outcome, 0)


def create_pairs_from_example(
    example: Dict[str, Any], example_id: str
) -> tuple[List[Dict], List[Dict], str]:
    """
    Process a single example and return (pairs, invalid_completions, outcome_category).
    outcome_category is one of: 'mixed', 'only_correct', 'only_fp', 'only_fn', 'only_invalid'
    """
    input_text = example["input"]
    ground_truth = example.get("ground_truth")
    completions = example.get("completions", [])

    # Classify all completions
    classified = []
    invalid = []

    for comp in completions:
        final_label = comp.get("final_label")
        outcome = classify_outcome(
            final_label, str(ground_truth) if ground_truth else ""
        )

        if outcome == "INVALID":
            invalid.append(
                {
                    "id": example_id,
                    "input": input_text,
                    "response": comp.get("response"),
                    "final_label": final_label,
                    "ground_truth": ground_truth,
                }
            )
        else:
            classified.append(
                {
                    "response": comp.get("response"),
                    "outcome": outcome,
                    "priority": get_priority(outcome),
                }
            )

    # Determine outcome category
    unique_outcomes = set(c["outcome"] for c in classified)
    valid_outcomes = {"CORRECT", "FP", "FN"}
    present_valid = unique_outcomes & valid_outcomes

    # Categorize the example based on outcomes
    if len(present_valid) == 0:
        # All outcomes are INVALID
        outcome_category = "only_invalid"
    elif len(present_valid) == 1:
        # Single valid outcome type
        single_outcome = list(present_valid)[0]
        if single_outcome == "CORRECT":
            outcome_category = "only_correct"
        elif single_outcome == "FP":
            outcome_category = "only_fp"
        elif single_outcome == "FN":
            outcome_category = "only_fn"
        else:
            outcome_category = "only_invalid"
    else:
        # Multiple different valid outcomes - can create pairs
        outcome_category = "mixed"

    # If not mixed (no pairs can be created), return empty pairs
    if outcome_category != "mixed":
        return [], invalid, outcome_category

    # Create all priority-based pairs for mixed outcomes
    pairs = []
    sorted_completions = sorted(classified, key=lambda x: x["priority"], reverse=True)

    for i, chosen in enumerate(sorted_completions):
        for rejected in sorted_completions[i + 1 :]:
            # Only create pairs where chosen has higher priority than rejected
            if chosen["priority"] > rejected["priority"]:
                pairs.append(
                    {
                        "prompt": input_text,
                        "chosen": chosen["response"],
                        "rejected": rejected["response"],
                    }
                )

    return pairs, invalid, outcome_category


def main():
    root = get_project_root()

    # Input path
    input_path = (
        root / "src" / "semevalpolar" / "finetuning" / "rlhf" / "inference_results.json"
    )

    # Output path
    output_path = (
        root / "src" / "semevalpolar" / "finetuning" / "rlhf" / "preference_pairs.json"
    )

    # Load data
    with open(input_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "dataset" in data:
        examples = data["dataset"]
    elif isinstance(data, list):
        examples = data
    else:
        examples = [data]

    # Process all examples
    all_pairs = []
    all_invalid = []
    skipped_examples = []  # Track CORRECT, FP, FN skips

    # Counters for outcome categories
    category_counts = {
        "mixed": 0,
        "only_correct": 0,
        "only_fp": 0,
        "only_fn": 0,
        "only_invalid": 0,
    }

    for idx, example in enumerate(examples, start=1):
        example_id = f"example_{idx:04d}"
        pairs, invalid, outcome_category = create_pairs_from_example(
            example, example_id
        )

        all_pairs.extend(pairs)
        all_invalid.extend(invalid)
        category_counts[outcome_category] += 1

        # Log skips (CORRECT, FP, FN only - no pairs created)
        if outcome_category in ["only_correct", "only_fp", "only_fn"]:
            skipped_examples.append(
                {
                    "id": example_id,
                    "input": example.get("input", ""),
                    "ground_truth": example.get("ground_truth"),
                    "outcome_category": outcome_category,
                    "completions_count": len(example.get("completions", [])),
                }
            )

    # Calculate statistics
    total_inputs = len(examples)
    mixed_count = category_counts["mixed"]

    # Build output
    output = {
        "pairs": all_pairs,
        "invalid_completions": all_invalid,
        "skipped_examples": skipped_examples,
        "stats": {
            "total_inputs": total_inputs,
            "mixed_outcomes_count": mixed_count,
            "only_correct_count": category_counts["only_correct"],
            "only_fp_count": category_counts["only_fp"],
            "only_fn_count": category_counts["only_fn"],
            "only_invalid_count": category_counts["only_invalid"],
            "invalid_completions_count": len(all_invalid),
            "pairs_created": len(all_pairs),
            "skipped_count": len(skipped_examples),
        },
    }

    # Save output
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")
    print(f"Total inputs: {total_inputs}")
    print(f"Mixed outcomes (with pairs): {mixed_count}")
    print(f"Only CORRECT: {category_counts['only_correct']}")
    print(f"Only FP: {category_counts['only_fp']}")
    print(f"Only FN: {category_counts['only_fn']}")
    print(f"Only INVALID: {category_counts['only_invalid']}")
    print(f"Invalid completions: {len(all_invalid)}")
    print(f"Pairs created: {len(all_pairs)}")
    print(f"Skipped examples (no pairs): {len(skipped_examples)}")


if __name__ == "__main__":
    main()
