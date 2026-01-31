import os
import json
import torch
import torch.nn.functional as F
from typing import List, Tuple
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

from semevalpolar.finetuning.instruct.finetune import load_config
from semevalpolar.utils import get_project_root


def predict_with_forward_pass(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict binary label using forward pass instead of generation.

    Args:
        model: The causal LM model (with LoRA)
        tokenizer: The model's tokenizer
        input_ids: Token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        threshold: Classification threshold for class 1

    Returns:
        predicted_labels: Binary predictions [batch_size]
        probabilities: P(y=1|x) [batch_size]
    """
    # Ensure inputs are on the same device as model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Get token IDs for '0' and '1'
    token_0_id = tokenizer.convert_tokens_to_ids("0")
    token_1_id = tokenizer.convert_tokens_to_ids("1")

    # Validate that tokens exist in vocabulary
    if token_0_id == tokenizer.unk_token_id or token_1_id == tokenizer.unk_token_id:
        raise ValueError("Tokens '0' or '1' not found in tokenizer vocabulary")

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

    # Find the position of the last non-padding token for each sequence
    seq_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]

    # Extract logits at the last token position for each sequence
    batch_size = input_ids.size(0)
    last_token_logits = logits[
        torch.arange(batch_size), seq_lengths
    ]  # [batch_size, vocab_size]

    # Extract logits for tokens '0' and '1'
    logits_0 = last_token_logits[:, token_0_id]  # [batch_size]
    logits_1 = last_token_logits[:, token_1_id]  # [batch_size]

    # Stack and apply softmax to get probabilities
    logits_0_1 = torch.stack([logits_0, logits_1], dim=-1)  # [batch_size, 2]
    probs = F.softmax(logits_0_1, dim=-1)  # [batch_size, 2]

    # Extract P(y=1|x) - probability for token '1'
    prob_class_1 = probs[:, 1]  # [batch_size]

    # Apply threshold to get predictions
    predicted_labels = (prob_class_1 >= threshold).long()  # [batch_size]

    return predicted_labels, prob_class_1


def generate_predictions_jsonl_forward_pass(
    inputs: List[str] | pd.Series,
    output_path: str = None,
    threshold: float = 0.5,
    batch_size: int = 8,
):
    """
    Runs POLAR inference using forward pass instead of generation and writes predictions.jsonl.

    Args:
        inputs: List of input texts or pandas Series
        output_path: Path to save predictions JSONL file
        threshold: Classification threshold for class 1
        batch_size: Batch size for processing
    """

    if output_path is None:
        output_path = os.path.join(get_project_root(), "predictions_forward_pass.jsonl")

    config = load_config()

    adapter_path = os.path.join(
        get_project_root(), "predictions", "instruct", "final_model_v2"
    )

    # Check if adapter exists
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Model adapter not found at {adapter_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert inputs to list if it's a pandas Series
    if isinstance(inputs, pd.Series):
        inputs = inputs.tolist()

    total_samples = len(inputs)
    processed_samples = 0

    with open(output_path, "w") as f:
        # Process in batches
        for i in tqdm(
            range(0, len(inputs), batch_size), desc="Running inference", unit="batch"
        ):
            batch_texts = inputs[i : i + batch_size]

            # Create prompts in the same format as training
            prompts = []
            for text in batch_texts:
                prompt = f"""Input:
{text}

Reasoning:
"""
                prompts.append(prompt)

            # Tokenize batch
            encodings = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )

            # Run forward pass inference
            predicted_labels, probabilities = predict_with_forward_pass(
                model=model,
                tokenizer=tokenizer,
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                threshold=threshold,
            )

            # Write results
            for j, (text, pred_label, prob) in enumerate(
                zip(batch_texts, predicted_labels, probabilities)
            ):
                record = {
                    "input": text,
                    "predicted_label": pred_label.item(),
                    "probability_class_1": prob.item(),
                    "threshold_used": threshold,
                }
                f.write(json.dumps(record) + "\n")

            processed_samples += len(batch_texts)

    print(f"Predictions saved to {output_path}")
    print(
        f"Processed {processed_samples}/{total_samples} samples with threshold={threshold}"
    )


if __name__ == "__main__":
    # Example usage
    test_inputs = [
        "This is a test input.",
        "Another test input with different content.",
    ]

    generate_predictions_jsonl_forward_pass(
        inputs=test_inputs, threshold=0.5, batch_size=2
    )
