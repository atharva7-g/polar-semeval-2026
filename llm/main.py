from openai import OpenAI

import os
import pandas as pd
import json
from datetime import datetime, timezone
from llm.prompt_utils import get_prompt
from llm.data_utils import read_dataset, batch_df

client = OpenAI()

def create_response(prompt, model="gpt-5-mini"):
    response = client.responses.create(
        model=model, input=prompt
    )

    return response

def create_gen(data_path, batch_size=10):
    df = read_dataset(data_path)
    gen = batch_df(df, batch_size=batch_size)

    return gen

def test_run(batch: pd.DataFrame, column_name="text"):
    prompt = get_prompt(list(batch[column_name]))
    response = create_response(prompt)

    return response

if __name__ == '__main__':
    client = OpenAI()
    data_path = "data/dev_phase/subtask1/dev/eng.csv"
