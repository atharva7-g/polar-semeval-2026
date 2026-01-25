import re
import json
from typing import List, Dict


def extract_label_robust(text: str):
    """
    Robust label extractor for generative models.
    Returns 0 or 1 if found, else None.
    """

    # Preferred: explicit Final Answer slot
    m = re.search(r"Final Answer[^01]*([01])", text, re.DOTALL)
    if m:
        return int(m.group(1))

    # Fallback: first standalone digit
    m = re.search(r"\b([01])\b", text)
    if m:
        return int(m.group(1))

    return None


def evaluate_predictions(
    predictions: List[str],
    gold_labels: List[int],
) -> Dict[str, float]:
    """
    Evaluates generative model outputs in a failure-tolerant way.
    """

    assert len(predictions) == len(gold_labels)

    valid = 0
    correct = 0
    invalid = 0
    invalid_examples = []

    for i, (pred_text, gold) in enumerate(zip(predictions, gold_labels)):
        label = extract_label_robust(pred_text)

        if label is None:
            invalid += 1
            invalid_examples.append((i, pred_text))
            continue

        valid += 1
        if label == gold:
            correct += 1

    accuracy = correct / valid if valid > 0 else 0.0
    coverage = valid / len(predictions)

    return {
        "accuracy": accuracy,
        "coverage": coverage,
        "num_examples": len(predictions),
        "num_valid": valid,
        "num_invalid": invalid,
        "num_correct": correct,
        "invalid_examples": invalid_examples,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <predictions.jsonl> <gold.jsonl>")
        sys.exit(1)

    predictions_path = sys.argv[1]
    gold_path = sys.argv[2]

    # Load predictions
    with open(predictions_path, "r") as f:
        predictions = [json.loads(line)["prediction"] for line in f]

    # Load gold labels
    with open(gold_path, "r") as f:
        gold_labels = [json.loads(line)["label"] for line in f]

    results = evaluate_predictions(predictions, gold_labels)

    print("=== EVALUATION RESULTS ===")
    print(f"Accuracy (valid only): {results['accuracy']:.4f}")
    print(f"Coverage: {results['coverage']:.4f}")
    print(f"Total examples: {results['num_examples']}")
    print(f"Valid predictions: {results['num_valid']}")
    print(f"Invalid predictions: {results['num_invalid']}")

    if results["num_invalid"] > 0:
        print("\nExamples with invalid output:")
        for idx, text in results["invalid_examples"][:5]:
            print(f"\n--- Example {idx} ---")
            print(text)
