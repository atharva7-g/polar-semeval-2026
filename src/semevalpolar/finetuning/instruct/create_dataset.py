import json
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from semevalpolar.finetuning.instruct.dataset_utils import split_jsonl
from semevalpolar.finetuning.instruct.templates import build_example
from semevalpolar.llm.data_utils import read_dataset
from semevalpolar.utils import get_project_root
from semevalpolar.llm.main import create_response_from_prompt_file
import time
REQUEST_DELAY = 0


def format_without_reasoning(text: str, label: int) -> dict:
    return build_example(
        x=text,
        r="",
        y=str(label),
    )


def format_with_reasoning(text: str, label: int, prompt_path: str, model: str) -> str:
    response = create_response_from_prompt_file(
        template_path=prompt_path, 
        input_text=text,
        label=label,
        model=model,
    )
    reasoning = response.output_text

    return reasoning

def convert_csv_to_jsonl(
    input_csv: str,
    output_jsonl: str,
    prompt_path: str,
    with_reasoning: bool = False,
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

        records.append({"text": record})
        time.sleep(REQUEST_DELAY)

    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {output_jsonl}")
    return records


def create_dataset(
    prompt_path: str,
    input_csv: Optional[str] = None,
    output_jsonl: Optional[str] = None,
    output_dir: Optional[str] = None,
    with_reasoning: bool = True,
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
        output_dir = str(
            root
            / "src"
            / "semevalpolar"
            / "finetuning"
            / "instruct"
            / "data"
            / "test"
            / "splits"
        )

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
        dst_dir=Path(output_dir),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    print("\nDone!")


if __name__ == "__main__":
    prompt_path = str(
        get_project_root()
        / "src"
        / "semevalpolar"
        / "finetuning"
        / "instruct"
        / "prompt-reasoning-v3.txt"
    )
    create_dataset(prompt_path=prompt_path)
