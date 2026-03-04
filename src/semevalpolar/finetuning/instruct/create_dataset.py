import json
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from semevalpolar.finetuning.instruct.dataset_utils import split_jsonl
from semevalpolar.finetuning.instruct.templates import build_example
from semevalpolar.llm.data_utils import read_dataset
from semevalpolar.utils import get_project_root


def format_without_reasoning(text: str, label: int) -> dict:
    return build_example(
        x=text,
        r="",
        y=str(label),
    )


def format_with_reasoning(text: str, label: int, prompt_path: str, model: str) -> dict:
    from semevalpolar.llm.main import create_response_ollama

    prompt_template = Path(prompt_path).read_text(encoding="utf-8")
    prompt = prompt_template.format(input_statements=text, ground_truth={label})

    response = create_response_ollama(
        input_text=text,
        reasoning_text="",
        label=str(label),
        prompt_path=prompt_path,
        model=model,
    )
    reasoning = response.output_text

    return build_example(x=text, r=reasoning, y=str(label))


def convert_csv_to_jsonl(
    input_csv: str,
    output_jsonl: str,
    with_reasoning: bool = False,
    prompt_path: Optional[str] = None,
    model: str = "gemma3:12b",
    limit: Optional[int] = None,
) -> list:
    df = read_dataset(input_csv)

    if limit:
        df = df.head(limit)

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        text = row["text"]
        label = int(row["polarization"])

        if with_reasoning:
            record = format_with_reasoning(text, label, prompt_path, model)
        else:
            record = format_without_reasoning(text, label)

        records.append(record)

    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {output_jsonl}")
    return records


def create_dataset(
    input_csv: Optional[str] = None,
    output_jsonl: Optional[str] = None,
    output_dir: Optional[str] = None,
    with_reasoning: bool = True,
    prompt_path: Optional[str] = None,
    model: str = "gemma3:12b",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 125,
    limit: Optional[int] = None,
):
    root = get_project_root()

    if input_csv is None:
        input_csv = str(root / "data-public" / "test" / "eng.csv")

    if output_jsonl is None:
        output_jsonl = str(
            root
            / "src"
            / "semevalpolar"
            / "finetuning"
            / "instruct"
            / "data"
            / "test"
            / "dataset.jsonl"
        )

    if output_dir is None:
        output_dir = (
            root
            / "src"
            / "semevalpolar"
            / "finetuning"
            / "instruct"
            / "data"
            / "test"
            / "splits"
        )
    else:
        output_dir = Path(output_dir)

    if with_reasoning and prompt_path is None:
        prompt_path = str(
            root
            / "src"
            / "semevalpolar"
            / "finetuning"
            / "instruct"
            / "prompt-reasoning-v3.txt"
        )

    print("=" * 60)
    print("Dataset Creation")
    print("=" * 60)
    print(f"Input CSV: {input_csv}")
    print(f"Output JSONL: {output_jsonl}")
    print(f"Output splits: {output_dir}")
    print(f"Mode: {'with reasoning' if with_reasoning else 'without reasoning'}")
    if with_reasoning:
        print(f"Prompt: {prompt_path}")
        print(f"Model: {model}")
    print(f"Split: train={train_ratio}, val={val_ratio}, seed={seed}")
    print("=" * 60)

    convert_csv_to_jsonl(
        input_csv=input_csv,
        output_jsonl=output_jsonl,
        with_reasoning=with_reasoning,
        prompt_path=prompt_path,
        model=model,
        limit=limit,
    )

    print(f"\nSplitting into train/val/test...")
    split_jsonl(
        src_path=Path(output_jsonl),
        dst_dir=output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    print("\nDone!")


if __name__ == "__main__":
    create_dataset()
