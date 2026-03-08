import json
from pathlib import Path

import pandas as pd

from semevalpolar.utils import get_project_root
from semevalpolar.finetuning.instruct.templates import evaluate_metrics

data_path = Path(get_project_root()) / "data-public" / "test" / "eng.csv"
df = pd.read_csv(data_path)

texts = df["text"].tolist()
gold = df["polarization"].tolist()

pred_path = (
    Path(get_project_root())
    / "src"
    / "semevalpolar"
    / "finetuning"
    / "instruct"
    / "predictions"
    / "predictions_all_10.jsonl"
)

labels = []
pred_text = []

with pred_path.open() as f:
    for line in f:
        obj = json.loads(line)
        labels.append(obj["extracted_label"])
        pred_text.append(obj["prediction"])

comparison = pd.DataFrame({"text": texts, "prediction": labels, "gold": gold})

comparison.head()


print(evaluate_metrics(comparison, "gold", "prediction"))
