import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple


def load_dataset(input_path: Path) -> List[Dict[str, Any]]:
    with open(input_path, "r") as f:
        data = json.load(f)
    return data["dataset"]


def split_dataset(
    dataset: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    total = train_ratio + test_ratio + val_ratio
    if not 0.99 <= total <= 1.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    shuffled = dataset.copy()
    random.seed(seed)
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_test = int(n_total * test_ratio)

    train_set = shuffled[:n_train]
    test_set = shuffled[n_train : n_train + n_test]
    val_set = shuffled[n_train + n_test :]

    return train_set, val_set, test_set


def save_splits(
    train_set: List[Dict[str, Any]],
    test_set: List[Dict[str, Any]],
    val_set: List[Dict[str, Any]],
    output_dir: Path,
    prefix: str = "response_dict",
) -> Tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / f"{prefix}_train.json"
    test_path = output_dir / f"{prefix}_test.json"
    val_path = output_dir / f"{prefix}_val.json"

    with open(train_path, "w") as f:
        json.dump({"dataset": train_set}, f, indent=2)

    with open(test_path, "w") as f:
        json.dump({"dataset": test_set}, f, indent=2)

    with open(val_path, "w") as f:
        json.dump({"dataset": val_set}, f, indent=2)

    return train_path, val_path, test_path


def split_and_save(
    input_path: Path = Path("../instruct/response_dict.json"),
    output_dir: Path = Path("."),
    train_ratio: float = 0.8,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
    prefix: str = "response_dict",
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Tuple[Path, Path, Path],
]:
    dataset = load_dataset(input_path)
    train_set, val_set, test_set = split_dataset(
        dataset, train_ratio, test_ratio, val_ratio, seed
    )
    paths = save_splits(train_set, val_set, test_set, output_dir, prefix)
    return train_set, val_set, test_set, paths


def main():
    split_and_save(
        input_path=Path("../instruct/response_dict.json"),
        output_dir=Path("./data/"),
        train_ratio=0.8,
        test_ratio=0.1,
        val_ratio=0.1,
        seed=123,
        prefix="response_dict",
    )


if __name__ == "__main__":
    main()
