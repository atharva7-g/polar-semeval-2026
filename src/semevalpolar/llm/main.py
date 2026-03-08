import os

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path

from semevalpolar.finetuning.instruct.local_inference import (
    LocalResponse,
    LocalResponseUsage,
)

from semevalpolar.llm.data_utils import (
    read_dataset,
    batch_df,
    parse_predictions,
    create_submission,
)

from semevalpolar.llm.prompt_utils import build_prompt
import ollama

from semevalpolar.utils import get_project_root


LIMIT = None

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def create_response_from_prompt_file(
    template_path: str,
    input_text: str,
    label,
    model: str = "qwen/qwen3-max",
    encoding: str = "utf-8",
):
    template = Path(template_path).read_text(encoding=encoding)

    prompt = template.format(
        input_statements=input_text,
        ground_truth=label,
    )

    return create_response(prompt, model)


def create_response_ollama(
    input_text: str,
    reasoning_text: str,
    label: str,
    prompt_path: str,
    model: str = "gemma3:12b",
) -> LocalResponse:

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(
        INPUT_TEXT=input_text,
        REASONING_TEXT=reasoning_text,
        LABEL=label,
    )

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": 0.0},
    )

    usage = response.get("usage", {})

    return LocalResponse(
        output_text=response["response"],
        usage=LocalResponseUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            cost=0.0,
        ),
    )


def create_response(prompt, model="google/gemma-3-12b-it:free"):
    response = client.responses.create(model=model, input=prompt, temperature=0)
    return response


def create_gen(data_path, batch_size=10, randomize=True):

    df = read_dataset(data_path)

    if LIMIT is not None:
        df = df.head(LIMIT)

    gen = batch_df(df, batch_size=batch_size, randomize=randomize)

    return gen


def test_run(
    batch: pd.DataFrame,
    column_name="text",
    prompt_path=f"{get_project_root()}/src/semevalpolar/llm/prompt-short-ds-tighten.txt",
    model="qwen/qwen-2.5-7b-instruct",
):

    prompt = build_prompt(
        list(batch[column_name]),
        prompt_path=prompt_path,
    )

    response = create_response(prompt, model=model)

    return response


def pipeline(data_path, output_path="predictions/subtask_1/pred_eng.csv"):

    gen = create_gen(data_path, batch_size=2, randomize=False)
    generator_list = list(gen)

    predictions = []
    usages = []

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_cost = 0.0

    pbar = tqdm(generator_list, desc="Processing batches")

    for batch in pbar:
        response = test_run(batch)

        parsed = parse_predictions(response.output_text)

        if parsed is None or len(parsed) != len(batch):
            print("Invalid completion detected. Retrying batch...")
            response = test_run(batch)
            parsed = parse_predictions(response.output_text)

        if parsed is None or len(parsed) != len(batch):
            print("Retry failed. Filling batch with None.")
            parsed = [None] * len(batch)

        predictions.append(parsed)

        usage = getattr(response, "usage", None)
        usages.append(usage)

        if usage:
            input_tokens = getattr(usage, "prompt_tokens", None)
            if input_tokens is None:
                input_tokens = getattr(usage, "input_tokens", 0)

            output_tokens = getattr(usage, "completion_tokens", None)
            if output_tokens is None:
                output_tokens = getattr(usage, "output_tokens", 0)

            total = getattr(usage, "total_tokens", input_tokens + output_tokens)

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_tokens += total

            cost = input_tokens * (0.04 / 1e6) + output_tokens * (0.13 / 1e6)
            total_cost += cost

        pbar.set_postfix(
            {
                "in": total_input_tokens,
                "out": total_output_tokens,
                "tok": total_tokens,
                "cost": f"${total_cost:.4f}",
            }
        )

    flat = [x for sub in predictions for x in sub]

    flat_df = pd.DataFrame(flat, columns=["value"])

    flat_df.to_csv(output_path, index=False)

    print("\nUsage summary")
    print(f"Input tokens: {total_input_tokens}")
    print(f"Output tokens: {total_output_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Estimated cost: ${total_cost:.4f}")


if __name__ == "__main__":
    pipeline(
        data_path=f"{get_project_root()}/data-public/test/eng.csv",
        output_path=f"{get_project_root()}/predictions/zero_shot/qwen257b/eng.csv",
    )
