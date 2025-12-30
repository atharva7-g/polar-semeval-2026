import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def run_inference_on_df(
    df: pd.DataFrame,
    text_column: str,
    model_dir: str,
    batch_size: int = 32,
    device: str | None = None,
):
    """
    Runs sequence classification inference on a DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    text_column : str
        Column containing text.
    model_dir : str
        Directory created by `save_pretrained()`.
    batch_size : int
        Inference batch size.
    device : str | None
        'cuda', 'cpu', or None (auto-detect).

    Returns
    -------
    np.ndarray
        Predicted class indices.
    """

    assert os.path.isdir(model_dir), f"Invalid model directory: {model_dir}"
    assert text_column in df.columns, f"Missing column: {text_column}"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    ).to(device)

    model.eval()

    texts = df[text_column].astype(str).tolist()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(batch_preds.cpu().numpy())

    return np.array(predictions)
