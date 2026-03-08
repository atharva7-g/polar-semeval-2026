import os
import re
import json
import codecs
from typing import List

import pandas as pd
import torch

from semevalpolar.finetuning.instruct.predict import generate_predictions_jsonl
from semevalpolar.utils import get_project_root

ROOT = get_project_root()

VAL_PATH = os.path.join(
    ROOT,
    "src",
    "semevalpolar",
    "finetuning",
    "instruct",
    "data",
    "test",
    "dataset.jsonl",
)
PRED_PATH = os.path.join(ROOT, "predictions", "inference", "predictions_all_10.jsonl")

INPUT_RE = re.compile(r"Input:\s*(.*?)\s*Reasoning:", re.DOTALL)
FINAL_RE = re.compile(r"Final Answer[^01]*([01])")


def load_inputs(records: List[str]) -> List[str]:
    inputs = []
    for r in records:
        m = INPUT_RE.search(r)
        if not m:
            inputs.append("")
            continue
        decoded = codecs.decode(m.group(1), "unicode_escape")
        inputs.append(" ".join(decoded.split()))
    return inputs


def extract_gold_labels(records: List[str]) -> List[int]:
    labels = []
    for r in records:
        m = FINAL_RE.search(r)
        if not m:
            raise ValueError("Missing gold label in val.jsonl")
        labels.append(int(m.group(1)))
    return labels


def load_predictions(path: str) -> List[int]:
    preds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            label = obj.get("extracted_label")
            if label is None:
                m = FINAL_RE.search(obj.get("prediction", ""))
                if m is None:
                    raise ValueError("Missing prediction label")
                label = int(m.group(1))
            preds.append(int(label))
    return preds


# ---------------------------
# Main
# ---------------------------
def main():
    print("CUDA available:", torch.cuda.is_available())

    with open(VAL_PATH, "r", encoding="utf-8") as f:
        records = f.readlines()

    inputs = load_inputs(records)

    generate_predictions_jsonl(inputs)


if __name__ == "__main__":
    main()
