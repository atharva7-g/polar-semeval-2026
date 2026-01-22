#!/usr/bin/env python3

import os
import pandas as pd

from semevalpolar.finetuning.instruct.local_inference import run_local_hf
from semevalpolar.finetuning.instruct.reasoning_prompt import run_examples_with_tqdm
from semevalpolar.finetuning.instruct.templates import (
    parse_prompt,
    run,
    build_text,
)
from semevalpolar.llm.data_utils import read_dataset
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

    example = example.sample(len(example), random_state=42).reset_index(drop=True)

    response_dict = run_examples_with_tqdm(
        example_df=example,
        run_fn=run_local_hf,
        parse_fn=parse_prompt,
        build_text_fn=build_text,
        prompt_path="prompt-reasoning.txt",
        model="google/gemma-3-27b-it",
        limit=2,
    )

    print("Totals:")
    print(response_dict["totals"])

    print("\nSample output:")
    print(response_dict["dataset"][0])


if __name__ == "__main__":
    main()
