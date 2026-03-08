import argparse
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from semevalpolar.finetuning.rlhf.evaluate_dpo import (
    calculate_all_metrics,
    evaluate_dpo_predictions,
    print_results,
)
from semevalpolar.utils import get_project_root


@dataclass
class SFTPipelineConfig:
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    sft_model_path: str = field(
        default_factory=lambda: os.path.join(
            get_project_root(), "predictions", "instruct", "final_model"
        )
    )
    test_data_path: str = field(
        default_factory=lambda: os.path.join(
            get_project_root(), "data-public", "test", "eng.csv"
        )
    )
    text_column: str = "text"
    label_column: str = "polarization"
    max_new_tokens: int = 512
    output_dir: str = field(
        default_factory=lambda: os.path.join(
            get_project_root(), "predictions", "sft_eval"
        )
    )


def load_sft_model(model_path: str, base_model_name: str):
    """Load base model with SFT LoRA adapter."""
    print(f"Loading base model: {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    current_embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != current_embedding_size:
        print(f"Resizing embeddings from {current_embedding_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    print(f"Loading SFT adapter from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SFT adapter not found at {model_path}")

    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    print("SFT model loaded successfully.")
    return model, tokenizer


def extract_label(text: str) -> Optional[int]:
    """Extract final label (0 or 1) from model output."""
    match = re.search(r"Final label:\s*([01])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r"Final Answer:\s*([01])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r"Final Answer:\s*(yes|no)", text, re.IGNORECASE)
    if match:
        answer = match.group(1).lower()
        return 1 if answer == "yes" else 0

    match = re.search(r"\b([01])\b\s*$", text.strip())
    if match:
        return int(match.group(1))

    return None


def generate_prediction(
    model, tokenizer, text: str, max_new_tokens: int = 512
) -> tuple:
    """Generate prediction for a single text input."""
    prompt = f"""Input:
{text}

Reasoning:
"""

    enc = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_length = enc["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    full_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(full_output)

    label = extract_label(full_output)

    return full_output, label


def run_sft_inference(
    model,
    tokenizer,
    input_csv_path: str,
    text_column: str = "text",
    max_new_tokens: int = 512,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Run SFT model inference on a CSV file."""
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")

    df = pd.read_csv(input_csv_path)

    if text_column not in df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found. Available columns: {list(df.columns)}"
        )

    if limit is not None:
        df = df.head(limit)
        print(f"Limit set to {limit} samples")

    print(f"Running inference on {len(df)} samples...")
    predictions = []
    labels = []

    for text in tqdm(df[text_column], desc="Generating predictions"):
        full_output, label = generate_prediction(model, tokenizer, text, max_new_tokens)
        predictions.append(full_output)
        labels.append(label)

    df["prediction_text"] = predictions
    df["predicted_label"] = labels

    return df


class SFTPipeline:
    def __init__(self, config: SFTPipelineConfig):
        self.config = config

    def run(self):
        """Run the complete SFT evaluation pipeline."""
        print("=" * 60)
        print("SFT Model Evaluation Pipeline")
        print("=" * 60)
        print(f"Model: {self.config.sft_model_path}")
        print(f"Test data: {self.config.test_data_path}")
        print(f"Output dir: {self.config.output_dir}")

        model, tokenizer = load_sft_model(
            self.config.sft_model_path, self.config.base_model_name
        )

        df = run_sft_inference(
            model,
            tokenizer,
            self.config.test_data_path,
            self.config.text_column,
            self.config.max_new_tokens,
        )

        os.makedirs(self.config.output_dir, exist_ok=True)
        output_path = os.path.join(self.config.output_dir, "sft_predictions.csv")
        df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")

        predictions_path = output_path
        print(f"\nEvaluating predictions...")

        basic_metrics, cm, total_samples, y_true, y_pred = evaluate_dpo_predictions(
            predictions_path
        )
        full_metrics = calculate_all_metrics(y_true, y_pred)
        print_results(full_metrics, cm, total_samples)

        return full_metrics


def main():
    parser = argparse.ArgumentParser(description="SFT Model Evaluation Pipeline")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to SFT model (LoRA adapter)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test CSV file",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of column containing text to classify",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="polarization",
        help="Name of column containing ground truth labels",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process",
    )

    args = parser.parse_args()

    config = SFTPipelineConfig()

    if args.model_path:
        config.sft_model_path = args.model_path
    if args.test_data:
        config.test_data_path = args.test_data
    if args.text_column:
        config.text_column = args.text_column
    if args.label_column:
        config.label_column = args.label_column
    if args.max_new_tokens:
        config.max_new_tokens = args.max_new_tokens
    if args.output_dir:
        config.output_dir = args.output_dir

    if args.limit:
        original_test_path = config.test_data_path
        import tempfile

        df = pd.read_csv(config.test_data_path)
        df_limited = df.head(args.limit)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
            df_limited.to_csv(tmp.name, index=False)
            config.test_data_path = tmp.name

    pipeline = SFTPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
