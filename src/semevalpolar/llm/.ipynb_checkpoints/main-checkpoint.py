from openai import OpenAI

import os
import pandas as pd
import json
from tqdm import tqdm
from datetime import datetime, timezone
from semevalpolar.llm.prompt_utils import get_prompt, build_prompt
from semevalpolar.llm.data_utils import read_dataset, batch_df, parse_predictions, create_submission

# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.getenv("OPENROUTER_API_KEY"),
# )


def create_response(prompt, model):
    response = client.responses.create(
        model=model, input=prompt
    )

    return response

def create_gen(data_path, batch_size=10, randomize=True):
    df = read_dataset(data_path)
    gen = batch_df(df, batch_size=batch_size, randomize=randomize)

    return gen

def test_run(batch: pd.DataFrame, column_name="text", prompt_path="prompt-three-classes.txt", model="qwen/qwen3-max"):
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


if __name__ == '__main__':
    response = create_response("Hi. How are you?")
    print(response.output_text)

