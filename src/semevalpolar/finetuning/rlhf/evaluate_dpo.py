#!/usr/bin/env python3
"""
Evaluation script for DPO predictions.
Computes F1, accuracy, precision, recall from dpo_predictions.csv
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path


def evaluate_dpo_predictions(csv_path: str | Path) -> dict:
    """
    Evaluate DPO predictions from CSV file.

    Args:
        csv_path: Path to the dpo_predictions.csv file

    Returns:
        Dictionary containing evaluation metrics
    """
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Check for missing values and drop rows with NaN
    missing_polarization = df["polarization"].isna().sum()
    missing_predictions = df["predicted_label"].isna().sum()

    if missing_polarization > 0 or missing_predictions > 0:
        print(
            f"Warning: Dropping {missing_polarization} rows with missing polarization "
            f"and {missing_predictions} rows with missing predictions."
        )
        df = df.dropna(subset=["polarization", "predicted_label"])

    # Extract ground truth and predictions
    y_true = df["polarization"].astype(int)
    y_pred = df["predicted_label"].astype(int)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Count samples
    total_samples = len(df)

    return metrics, cm, total_samples, y_true, y_pred


def print_results(metrics: dict, cm: list, total_samples: int) -> None:
    """Print evaluation results in a formatted manner."""
    print("=" * 50)
    print("DPO PREDICTIONS EVALUATION RESULTS")
    print("=" * 50)
    print(f"\nTotal Samples: {total_samples}")

    print("\n" + "-" * 50)
    print("CONFUSION MATRIX")
    print("-" * 50)
    print("                 Predicted")
    print("                 0      1")
    print(
        f"Actual    0     {cm[0][0]:4d}   {cm[0][1]:4d}   (TN={cm[0][0]}, FP={cm[0][1]})"
    )
    print(
        f"          1     {cm[1][0]:4d}   {cm[1][1]:4d}   (FN={cm[1][0]}, TP={cm[1][1]})"
    )

    print("\n" + "-" * 50)
    print("CLASSIFICATION METRICS")
    print("-" * 50)
    print(
        f"Accuracy:     {metrics['accuracy']:.4f}  ({metrics['accuracy'] * 100:.2f}%)"
    )
    print(f"Precision:    {metrics['precision']:.4f}  (Class 1: Polarized)")
    print(f"Recall:       {metrics['recall']:.4f}  (Class 1: Polarized)")
    print(f"F1-Score:     {metrics['f1_score']:.4f}  (Class 1: Polarized)")

    print("\n" + "-" * 50)
    print("DETAILED CLASSIFICATION REPORT")
    print("-" * 50)
    # Calculate class-specific metrics
    print("\nPer-Class Metrics:")
    print(f"Class 0 (Not Polarized):")
    print(f"  Precision: {metrics['precision_neg']:.4f}")
    print(f"  Recall:    {metrics['recall_neg']:.4f}")
    print(f"  F1-Score:  {metrics['f1_neg']:.4f}")
    print(f"  Support:   {metrics['support_neg']}")
    print(f"\nClass 1 (Polarized):")
    print(f"  Precision: {metrics['precision_pos']:.4f}")
    print(f"  Recall:    {metrics['recall_pos']:.4f}")
    print(f"  F1-Score:  {metrics['f1_pos']:.4f}")
    print(f"  Support:   {metrics['support_pos']}")
    print("\nMacro Average:")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall:    {metrics['recall_macro']:.4f}")
    print(f"  F1-Score:  {metrics['f1_macro']:.4f}")

    print("\n" + "=" * 50)


def calculate_all_metrics(y_true, y_pred) -> dict:
    """Calculate comprehensive metrics for both classes and averages."""
    from sklearn.metrics import precision_recall_fscore_support

    # Get per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )

    # Macro average
    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_macro = f1.mean()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision[1],  # Class 1
        "recall": recall[1],  # Class 1
        "f1_score": f1[1],  # Class 1
        "precision_neg": precision[0],  # Class 0
        "recall_neg": recall[0],  # Class 0
        "f1_neg": f1[0],  # Class 0
        "precision_pos": precision[1],  # Class 1
        "recall_pos": recall[1],  # Class 1
        "f1_pos": f1[1],  # Class 1
        "support_neg": support[0],
        "support_pos": support[1],
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }

    return metrics


def main():
    """Main function to run evaluation."""
    # Path to the CSV file
    csv_path = (
        Path(__file__).parent
        / "dpo_predictions"
        / "improved_but_close"
        / "dpo_predictions_v8_126.csv"
    )

    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Evaluate
    basic_metrics, cm, total_samples, y_true, y_pred = evaluate_dpo_predictions(
        csv_path
    )

    # Calculate comprehensive metrics
    full_metrics = calculate_all_metrics(y_true, y_pred)

    # Print results
    print_results(full_metrics, cm, total_samples)

    # Also print scikit-learn's classification report
    print("\n" + "=" * 50)
    print("SKLEARN CLASSIFICATION REPORT")
    print("=" * 50)
    print(
        classification_report(
            y_true, y_pred, target_names=["Not Polarized (0)", "Polarized (1)"]
        )
    )


if __name__ == "__main__":
    main()
