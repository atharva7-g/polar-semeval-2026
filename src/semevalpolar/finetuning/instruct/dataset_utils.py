import json
import random
from pathlib import Path

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
    n_val   = int(n_total * val_ratio)

    splits = {
        "train": lines[:n_train],
        "val"  : lines[n_train:n_train + n_val],
        "test" : lines[n_train + n_val:],
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


if __name__ == "__main__":
    src = Path("data/dataset.jsonl")          # original file
    out_dir = Path("data/splits")           # where train/val/test will be written
    split_jsonl(src, out_dir, train_ratio=0.8, val_ratio=0.1, seed=123)
