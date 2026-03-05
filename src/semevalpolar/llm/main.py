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
    # load prompt template
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
        options={
            "temperature": 0.0,
        },
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
    response = client.responses.create(model=model, input=prompt)

    return response


def create_gen(data_path, batch_size=10, randomize=True):
    df = read_dataset(data_path)
    gen = batch_df(df, batch_size=batch_size, randomize=randomize)

    return gen


def test_run(
    batch: pd.DataFrame,
    column_name="text",
    prompt_path="prompt-three-classes.txt",
    model="qwen/qwen3-max",
):
    prompt = build_prompt(list(batch[column_name]), prompt_path=prompt_path)
    response = create_response(prompt, model=model)

    return response


def pipeline(data_path, output_path="predictions/subtask_1/pred_eng.csv"):
    gen = create_gen(data_path, batch_size=10, randomize=False)

    generator_list = list(gen)
    df = read_dataset(data_path)

    predictions = []
    usages = []

    for batch in tqdm(generator_list):
        response = test_run(batch)
        predictions.append(parse_predictions(response.output_text))
        usages.append(response.usage)

    flat = [x for sub in predictions for x in sub]
    submission = create_submission(df, flat)
    submission.to_csv(output_path)


if __name__ == "__main__":
    response = create_response("Hi. How are you?")
    print(response.output_text)
