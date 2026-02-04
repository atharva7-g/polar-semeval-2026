"""
DPO (Direct Preference Optimization) training script for binary classification.

Loads SFT LoRA checkpoint as both policy and reference model.
Uses low beta to allow deviation from reference for FN-heavy optimization.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOConfig, DPOTrainer

from semevalpolar.utils import get_project_root


@dataclass
class DPOTrainingConfig:
    """Configuration for DPO training."""

    # Model paths
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    sft_adapter_path: str = field(
        default_factory=lambda: os.path.join(
            get_project_root(), "predictions", "instruct", "final_model"
        )
    )

    # Data paths
    preference_data_path: str = field(
        default_factory=lambda: os.path.join(
            get_project_root(),
            "src",
            "semevalpolar",
            "finetuning",
            "rlhf",
            "dual_prompt",
            "preference_pairs_cleaned.json",
        )
    )
    output_dir: str = field(
        default_factory=lambda: os.path.join(
            get_project_root(), "predictions", "instruct", "dpo_model"
        )
    )

    # DPO-specific hyperparameters
    beta: float = (
        0.1  # Low beta allows more deviation from reference for FN-heavy tasks
    )

    # Training hyperparameters (conservative for stability)
    learning_rate: float = 5e-6  # Small LR to avoid catastrophic forgetting
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Effective batch size = 8
    max_length: int = 1024
    max_prompt_length: int = 512

    # System settings
    dtype: str = "bf16"
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2


def load_preference_dataset(config: DPOTrainingConfig) -> Dataset:
    """Load and format preference pairs for DPO training."""

    with open(config.preference_data_path, "r") as f:
        data = json.load(f)

    pairs = data.get("pairs", [])

    formatted_data = []
    for pair in pairs:
        formatted_data.append(
            {
                "prompt": pair["input"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            }
        )

    dataset = Dataset.from_list(formatted_data)
    print(f"Loaded {len(dataset)} preference pairs from {config.preference_data_path}")

    return dataset


def load_model_and_tokenizer(config: DPOTrainingConfig):
    """Load base model with SFT LoRA adapter."""

    print(f"Loading base model: {config.base_model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    # Qwen tokenizer setup (consistent with SFT training)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        torch_dtype=torch.bfloat16 if config.dtype == "bf16" else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Resize embeddings if needed (consistent with SFT)
    current_embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != current_embedding_size:
        print(f"Resizing embeddings from {current_embedding_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    print(f"Loading SFT adapter from: {config.sft_adapter_path}")
    if not os.path.exists(config.sft_adapter_path):
        raise FileNotFoundError(f"SFT adapter not found at {config.sft_adapter_path}")

    model = PeftModel.from_pretrained(model, config.sft_adapter_path)

    model.enable_input_require_grads()

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    print(f"Model loaded successfully. Trainable parameters:")
    model.print_trainable_parameters()

    return model, tokenizer


def create_reference_model(config: DPOTrainingConfig, tokenizer):
    """Load reference model with SFT adapter and freeze it."""

    print(f"Loading reference base model: {config.base_model_name}")

    # Load base model (same as policy)
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        torch_dtype=torch.bfloat16 if config.dtype == "bf16" else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Resize embeddings if needed
    current_embedding_size = ref_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != current_embedding_size:
        ref_model.resize_token_embeddings(len(tokenizer))

    # Load SFT LoRA adapter (same as policy)
    print(f"Loading SFT adapter for reference: {config.sft_adapter_path}")
    ref_model = PeftModel.from_pretrained(ref_model, config.sft_adapter_path)

    # Freeze all parameters
    for param in ref_model.parameters():
        param.requires_grad = False

    ref_model.eval()

    print("Reference model loaded and frozen.")

    return ref_model


def setup_dpo_trainer(
    model,
    ref_model,
    tokenizer,
    train_dataset: Dataset,
    config: DPOTrainingConfig,
) -> DPOTrainer:
    """Configure and create DPO trainer with conservative hyperparameters."""

    training_args = DPOConfig(
        output_dir=config.output_dir,
        # Low beta allows policy to deviate from reference for FN-heavy tasks
        beta=config.beta,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_dir=os.path.join(config.output_dir, "logs"),
        optim="adamw_torch",
        bf16=config.dtype == "bf16",
        fp16=config.dtype == "fp16",
        gradient_checkpointing=True,
        # No evaluation during DPO training
        eval_strategy="no",
        remove_unused_columns=False,
        push_to_hub=False,
        # Disable built-in wandb/tensorboard to avoid conflicts
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    return trainer


def main():
    """Main DPO training pipeline."""

    # Initialize configuration
    config = DPOTrainingConfig()

    print("=" * 60)
    print("DPO Training Configuration:")
    print(
        f"  Beta: {config.beta} (low beta = more deviation from reference, good for FN-heavy tasks)"
    )
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(
        f"  Batch size (effective): {config.per_device_train_batch_size * config.gradient_accumulation_steps}"
    )
    print(f"  SFT adapter: {config.sft_adapter_path}")
    print(f"  Output: {config.output_dir}")
    print("=" * 60)

    # Load dataset
    print("\nLoading preference dataset...")
    train_dataset = load_preference_dataset(config)

    # Load model and tokenizer
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(config)

    # Create reference model (frozen copy)
    print("\nCreating reference model (frozen)...")
    ref_model = create_reference_model(config, tokenizer)

    # Setup trainer
    print("\nSetting up DPO trainer...")
    trainer = setup_dpo_trainer(model, ref_model, tokenizer, train_dataset, config)

    # Train
    print("\nStarting DPO training...")
    print(
        f"Training on {len(train_dataset)} preference pairs for {config.num_train_epochs} epochs"
    )
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    os.makedirs(config.output_dir, exist_ok=True)

    # Save the trained model (includes the updated LoRA weights)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Save config for reproducibility
    config_dict = {
        "base_model_name": config.base_model_name,
        "sft_adapter_path": config.sft_adapter_path,
        "beta": config.beta,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
    }

    with open(os.path.join(config.output_dir, "dpo_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nTraining complete! Model saved to: {config.output_dir}")
    print(f"Config saved to: {os.path.join(config.output_dir, 'dpo_config.json')}")


if __name__ == "__main__":
    main()
