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
) -> tuple[List[Dict], List[Dict], Optional[Dict]]:
    """
    Process a single example and return (pairs, invalid_completions, skipped_info).
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

    # Check if we have enough valid outcomes to make pairs
    unique_outcomes = set(c["outcome"] for c in classified)
    valid_outcomes = {"CORRECT", "FP", "FN"}
    present_valid = unique_outcomes & valid_outcomes

    if len(present_valid) < 2:
        reason = (
            f"only {', '.join(sorted(present_valid))}"
            if present_valid
            else "no valid outcomes"
        )
        skipped_info = {
            "id": example_id,
            "input": input_text,
            "reason": reason,
            "unique_outcomes": list(present_valid),
        }
        return [], invalid, skipped_info

    # Create all priority-based pairs
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

    return pairs, invalid, None


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
    all_skipped = []

    for idx, example in enumerate(examples, start=1):
        example_id = f"example_{idx:04d}"
        pairs, invalid, skipped = create_pairs_from_example(example, example_id)

        all_pairs.extend(pairs)
        all_invalid.extend(invalid)
        if skipped:
            all_skipped.append(skipped)

    # Calculate statistics
    total_inputs = len(examples)
    skipped_count = len(all_skipped)
    valid_inputs = total_inputs - skipped_count

    # Build output
    output = {
        "pairs": all_pairs,
        "invalid_completions": all_invalid,
        "skipped_inputs": all_skipped,
        "stats": {
            "total_inputs": total_inputs,
            "valid_inputs": valid_inputs,
            "skipped_count": skipped_count,
            "invalid_completions_count": len(all_invalid),
            "pairs_created": len(all_pairs),
        },
    }

    # Save output
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")
    print(f"Total inputs: {total_inputs}")
    print(f"Valid inputs (with pairs): {valid_inputs}")
    print(f"Skipped inputs: {skipped_count}")
    print(f"Invalid completions: {len(all_invalid)}")
    print(f"Pairs created: {len(all_pairs)}")


if __name__ == "__main__":
    main()
