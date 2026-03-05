#!/usr/bin/env python3

import os
import json
import pandas as pd

from semevalpolar.finetuning.instruct.local_inference import run_local_ollama
from semevalpolar.finetuning.instruct.reasoning_prompt import run_examples_with_tqdm
from semevalpolar.finetuning.instruct.templates import (
    parse_prompt,
    run,
    build_text,
    parse_prompt_structured,
)
from semevalpolar.llm.data_utils import read_dataset
from semevalpolar.llm.main import create_response_ollama
from semevalpolar.utils import get_project_root


def main():
    data_path = os.path.join(
        get_project_root(),
        "data",
        "dev_phase",
        "subtask1",
        "train",
        "eng.csv",
    )

    df = read_dataset(data_path)

    example = pd.DataFrame(
        {
            "text": df["text"],
            "ground_truth": df["polarization"],
        }
    )

    example = example.sample(frac=1, random_state=42).reset_index(drop=True)

    response_dict = run_examples_with_tqdm(
        example_df=example,
        run_fn=run_local_ollama,
        parse_fn=parse_prompt_structured,
        build_text_fn=build_text,
        prompt_path=f"{get_project_root()}/src/semevalpolar/finetuning/instruct/prompt-reasoning-v3.txt",
        model="gemma3:27b",
    )

    output_path = (
        f"{get_project_root()}/src/semevalpolar/finetuning/instruct/response_dict.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(response_dict, f, indent=2, ensure_ascii=False)


import json
from tqdm import tqdm
from datetime import datetime


def run_ollama_on_dataset(
    input_file_path: str,
    prompt_path: str,
    output_file_path: str,
) -> None:
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for item in tqdm(data["dataset"], desc="Processing responses"):
        text = item["input"]
        reasoning = item["reasoning"]
        label = int(item["final answer (polarization)"])

        response = create_response_ollama(
            prompt_path=prompt_path,
            input_text=text,
            reasoning_text=reasoning,
            label=str(label),
        )

        results.append(
            {
                "input": text,
                "reasoning": reasoning,
                "label": label,
                "model_output": response,
            }
        )

    output_payload = {
        "meta": {
            "prompt_path": prompt_path,
            "num_samples": len(results),
            "timestamp": datetime.utcnow().isoformat(),
        },
        "results": results,
    }

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    instruct_folder = f"{get_project_root()}/src/semevalpolar/finetuning/instruct"

    run_ollama_on_dataset(
        f"{instruct_folder}/response_dict.json",
        f"{instruct_folder}/prompt-ambiguous.txt",
        f"{instruct_folder}/ambiguity.json",
    )
