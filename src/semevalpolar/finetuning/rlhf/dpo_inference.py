"""DPO model inference script for binary classification.

Provides function-based API for running inference with trained DPO model.
"""

import os
import re
from typing import Optional

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from semevalpolar.utils import get_project_root


# Default paths
DPO_MODEL_PATH = os.path.join(
    get_project_root(), "predictions", "instruct", "dpo_model_v1"
)
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = os.path.join(
    get_project_root(), "src", "semevalpolar", "finetuning", "rlhf", "dpo_predictions"
)


def load_dpo_model():
    """Load base model with DPO LoRA adapter."""
    print(f"Loading base model: {BASE_MODEL_NAME}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        use_fast=True,
        trust_remote_code=True,
    )

    # Qwen tokenizer setup
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Resize embeddings if needed
    current_embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != current_embedding_size:
        print(f"Resizing embeddings from {current_embedding_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # Load DPO adapter
    print(f"Loading DPO adapter from: {DPO_MODEL_PATH}")
    if not os.path.exists(DPO_MODEL_PATH):
        raise FileNotFoundError(f"DPO adapter not found at {DPO_MODEL_PATH}")

    model = PeftModel.from_pretrained(model, DPO_MODEL_PATH)
    model.eval()

    # Setup model config
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    print("DPO model loaded successfully.")
    return model, tokenizer


def extract_label(text: str) -> Optional[int]:
    """Extract final label (0 or 1) from model output.

    Handles multiple formats:
    - "Final label: 0/1" (DPO training format)
    - "Final Answer: 0/1" (SFT format)
    - "Final Answer: yes/no" (alternative format)
    """
    # Look for "Final label: 0" or "Final label: 1"
    match = re.search(r"Final label:\s*([01])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Look for "Final Answer: 0" or "Final Answer: 1"
    match = re.search(r"Final Answer:\s*([01])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Look for "Final Answer: yes" (treat as 1) or "Final Answer: no" (treat as 0)
    match = re.search(r"Final Answer:\s*(yes|no)", text, re.IGNORECASE)
    if match:
        answer = match.group(1).lower()
        return 1 if answer == "yes" else 0

    # Look for standalone 0 or 1 at the end of the text
    match = re.search(r"\b([01])\b\s*$", text.strip())
    if match:
        return int(match.group(1))

    return None


def generate_prediction(
    model, tokenizer, text: str, max_new_tokens: int = 256
) -> tuple:
    """Generate prediction for a single text input.

    Returns:
        tuple: (full_output_text, extracted_label)
    """
    # Use hybrid format with explicit instructions for output structure
    prompt = f"""Input:
{text}

Reasoning:
"""

    # Encode and generate
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens (exclude input prompt)
    input_length = enc["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    full_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Extract label
    label = extract_label(full_output)

    return full_output, label


def run_dpo_inference(
    input_csv_path: str,
    text_column: str = "text",
    output_filename: str = "dpo_predictions.csv",
    max_new_tokens: int = 256,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Run DPO model inference on a CSV file.

    Args:
        input_csv_path: Path to input CSV file containing text to classify
        text_column: Name of column containing text to classify (default: "text")
        output_filename: Name for output CSV (saved in rlhf directory)
        max_new_tokens: Maximum tokens to generate for each prediction
        limit: Maximum number of samples to process (default: None = process all)

    Returns:
        DataFrame with original columns plus 'prediction_text' and 'predicted_label'

    Raises:
        FileNotFoundError: If input CSV doesn't exist
        ValueError: If text_column not found in CSV
    """
    # Load input data
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")

    df = pd.read_csv(input_csv_path)

    if text_column not in df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found. Available columns: {list(df.columns)}"
        )

    # Limit the number of samples if specified
    if limit is not None:
        df = df.head(limit)
        print(f"Limit set to {limit} samples")

    # Load model
    print("Loading DPO model...")
    model, tokenizer = load_dpo_model()

    # Run inference
    print(f"Running inference on {len(df)} samples...")
    predictions = []
    labels = []

    for text in tqdm(df[text_column], desc="Generating predictions"):
        full_output, label = generate_prediction(model, tokenizer, text, max_new_tokens)
        predictions.append(full_output)
        labels.append(label)

    # Add results to dataframe
    df["prediction_text"] = predictions
    df["predicted_label"] = labels

    # Add metadata about training dataset
    df["training_dataset"] = "dpo_preference_pairs.json"

    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    df.to_csv(output_path, index=False)

    print(f"\nPredictions saved to: {output_path}")
    print(f"Training dataset: dpo_preference_pairs.json")
    print(f"Total samples: {len(df)}")
    print(f"Valid predictions: {sum(1 for l in labels if l is not None)}")
    print(f"Invalid predictions: {sum(1 for l in labels if l is None)}")

    return df


if __name__ == "__main__":
    run_dpo_inference(
        os.path.join(
            get_project_root(), "data", "test_phase", "subtask1", "dev", "eng.csv"
        ),
        limit=None,
    )
