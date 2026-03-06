import os
import re
import json
import torch
from typing import List

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import pandas as pd
from semevalpolar.finetuning.instruct.finetune import load_config
from semevalpolar.utils import get_project_root

from tqdm import tqdm


def generate_predictions_jsonl(
    inputs: List[str] | pd.Series,
    output_path: str = os.path.join(get_project_root(), "predictions.jsonl"),
):
    """
    Runs POLAR inference on a list of input texts and writes predictions.jsonl.
    """

    def extract_label(text: str):
        m = re.search(r"Final Answer[^01]*([01])", text, re.DOTALL)
        if m:
            return int(m.group(1))
        return None

    config = load_config()

    adapter_path = os.path.join(
        get_project_root(),
        "predictions",
        "instruct",
        "sft_model_all_10"
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for text in tqdm(inputs, desc="Running inference", unit="sample"):
            prompt = f"""Input:
{text}

Reasoning:
"""

            enc = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **enc,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

            record = {
                "input": text,
                "prediction": decoded,
                "extracted_label": extract_label(decoded),
            }

            f.write(json.dumps(record) + "\n")
