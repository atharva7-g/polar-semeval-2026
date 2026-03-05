import json
import random
import os
import pandas as pd
from pathlib import Path
from semevalpolar.utils import get_project_root
import re


def split_jsonl(
    src_path: Path,
    dst_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Reads *src_path* (one JSON object per line) and writes three files:
            train.jsonl, val.jsonl, test.jsonl
    Ratios must sum ≤ 1.0; the remainder goes to the test split.
    """
    # Load all lines
    with src_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # Shuffle reproducibly
    random.seed(seed)
    random.shuffle(lines)

    n_total = len(lines)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": lines[:n_train],
        "val": lines[n_train : n_train + n_val],
        "test": lines[n_train + n_val :],
    }

    dst_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_lines in splits.items():
        out_path = dst_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as out_f:
            out_f.writelines(split_lines)

    print(
        f"Split {n_total} records → "
        f"{len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test"
    )


# def extract_label(text: str) -> int:
# 	m = re.search(
# 		r"Final Answer:\\n[01]",
# 		text,
# 	)
# 	if not m:
# 		raise ValueError("Polarization label not found.")
# 	return int(m.group(0).replace("Final Answer:\\n", ""))


def extract_label(text: str):
    m = re.search(r"Final Answer[^01]*([01])", text, re.DOTALL)
    if m:
        return int(m.group(1))
    return None


def create_balanced_csv(project_root, data_path, polarization_column, output_path):
    """
    Creates a balanced CSV file by sampling an equal number of polarized and non-polarized examples.

    Args:
    - project_root (str): The root directory of the project.
    - data_path (str): The path to the input CSV file.
    - polarization_column (str): The name of the column indicating polarization.
    - output_path (str): The path where the balanced CSV file will be saved.

    Returns:
    - None
    """

    try:
        data_file_path = Path(os.path.join(project_root, data_path))
        dev_df = pd.read_csv(data_file_path)

        # Check if polarization column exists
        if polarization_column not in dev_df.columns:
            raise ValueError(
                f"The '{polarization_column}' column does not exist in the DataFrame."
            )

        # Shuffle data
        dev_df = dev_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Filter data
        dev_df_polar = dev_df[dev_df[polarization_column] == 1]
        dev_df_not_polar = dev_df[dev_df[polarization_column] == 0]

        print(f"Number of polarized samples: {len(dev_df_polar)}")
        print(f"Number of non-polarized samples: {len(dev_df_not_polar)}")

        # Balance data
        if len(dev_df_not_polar) < len(dev_df_polar):
            print(
                f"Not enough non-polarized samples. Using all {len(dev_df_not_polar)} samples."
            )
            balanced_df = pd.concat(
                [
                    dev_df_polar.sample(n=len(dev_df_not_polar), random_state=42),
                    dev_df_not_polar,
                ],
                ignore_index=True,
            )
        else:
            dev_df_not_polar = dev_df_not_polar.sample(
                n=len(dev_df_polar), random_state=42
            )
            balanced_df = pd.concat([dev_df_polar, dev_df_not_polar], ignore_index=True)

        # Save balanced data
        balanced_df.to_csv(output_path, index=False)
        print(f"Balanced data saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    src = Path("data/dataset.jsonl")  # original file
    out_dir = Path("data/splits")  # where train/val/test will be written
    split_jsonl(src, out_dir, train_ratio=0.8, val_ratio=0.1, seed=123)
