import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from semevalpolar.utils import get_project_root


@dataclass
class DPOPipelineConfig:
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    sft_output_dir: str = field(
        default_factory=lambda: os.path.join(
            get_project_root(), "predictions", "dpo-pipeline", "sft_qwen_model"
        )
    )
    dpo_output_dir: str = field(
        default_factory=lambda: os.path.join(
            get_project_root(), "predictions", "dpo-pipeline", "dpo_qwen_model"
        )
    )

    sft_adapter_path: str = field(
        default_factory=lambda: os.path.join(
            get_project_root(), "predictions", "instruct", "final_model"
        )
    )

    train_data_path: str = field(
        default_factory=lambda: os.path.join(
            get_project_root(),
            "src",
            "semevalpolar",
            "finetuning",
            "instruct",
            "data",
            "archive",
            "splits",
            "train.jsonl",
        )
    )
    preference_pairs_path: str = field(
        default_factory=lambda: os.path.join(
            get_project_root(),
            "src",
            "semevalpolar",
            "finetuning",
            "rlhf",
            "archive",
            "predictions",
            "v1",
            "preference_pairs_backup.json",
        )
    )

    sft_num_epochs: int = 3
    sft_learning_rate: float = 5e-5
    sft_train_batch_size: int = 1
    sft_gradient_accumulation_steps: int = 8

    dpo_num_epochs: int = 2
    dpo_learning_rate: float = 5e-6
    dpo_beta: float = 0.1
    dpo_train_batch_size: int = 1
    dpo_gradient_accumulation_steps: int = 8

    max_length: int = 1024

    test_data_path: str = field(
        default_factory=lambda: os.path.join(
            get_project_root(), "data", "test_phase", "subtask1", "dev", "eng.csv"
        )
    )


class DPOPipeline:
    def __init__(self, config: DPOPipelineConfig = None):
        self.config = config or DPOPipelineConfig()

    def step1_train_sft(self):
        """Step 1: Train SFT model on Qwen."""
        print("=" * 60)
        print("STEP 1: Training SFT Model on Qwen")
        print("=" * 60)

        from semevalpolar.finetuning.instruct.finetune import (
            TrainingConfig,
            TrainingPipeline,
            load_config,
        )
        from transformers import AutoTokenizer
        from datasets import load_dataset
        from semevalpolar.finetuning.instruct.dataset import PolarDataset

        config = load_config()

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            use_fast=True,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        train_data = load_dataset("json", data_files=self.config.train_data_path)[
            "train"
        ]
        texts = train_data["text"]

        train_dataset = PolarDataset(
            texts=texts,
            config=config,
            tokenizer=tokenizer,
        )

        pipeline = TrainingPipeline(config=config, tokenizer=tokenizer)
        pipeline.run(train_dataset)

        sft_final_path = os.path.join(config.output_dir, "final_model")
        self.config.adapter_path = sft_final_path

        print(f"\nSFT training complete! Model saved to: {sft_final_path}")
        return sft_final_path

    def step2_train_dpo(self):
        """Step 2: Train DPO model on Qwen."""
        print("\n" + "=" * 60)
        print("STEP 2: Training DPO Model on Qwen")
        print("=" * 60)

        from semevalpolar.finetuning.rlhf.dpo_train import (
            DPOTrainingConfig,
            load_preference_dataset,
            load_model_and_tokenizer,
            create_reference_model,
            setup_dpo_trainer,
        )

        config = DPOTrainingConfig()
        config.base_model_name = self.config.base_model_name
        config.sft_adapter_path = self.config.sft_adapter_path
        config.output_dir = self.config.dpo_output_dir
        config.preference_data_path = self.config.preference_pairs_path
        config.num_train_epochs = self.config.dpo_num_epochs
        config.learning_rate = self.config.dpo_learning_rate
        config.beta = self.config.dpo_beta
        config.per_device_train_batch_size = self.config.dpo_train_batch_size
        config.gradient_accumulation_steps = self.config.dpo_gradient_accumulation_steps
        config.max_length = self.config.max_length

        print(f"Base model: {config.base_model_name}")
        print(f"SFT adapter: {config.sft_adapter_path}")
        print(f"Preference pairs: {config.preference_data_path}")
        print(f"Output: {config.output_dir}")
        print(
            f"Beta: {config.beta}, LR: {config.learning_rate}, Epochs: {config.num_train_epochs}"
        )

        train_dataset = load_preference_dataset(config)
        model, tokenizer = load_model_and_tokenizer(config)
        ref_model = create_reference_model(config, tokenizer)
        trainer = setup_dpo_trainer(model, ref_model, tokenizer, train_dataset, config)

        print(f"\nStarting DPO training on {len(train_dataset)} preference pairs...")
        trainer.train()

        print(f"\nDPO training complete! Model saved to: {config.output_dir}")
        return config.output_dir

    def step3_evaluate(self):
        """Step 3: Run inference and evaluate DPO model."""
        print("\n" + "=" * 60)
        print("STEP 3: Evaluating DPO Model")
        print("=" * 60)

        from semevalpolar.finetuning.rlhf.dpo_inference import (
            run_dpo_inference,
            DPO_MODEL_PATH,
        )
        from semevalpolar.finetuning.rlhf.evaluate_dpo import (
            evaluate_dpo_predictions,
            calculate_all_metrics,
            print_results,
        )
        import pandas as pd

        original_dpo_path = DPO_MODEL_PATH

        import semevalpolar.finetuning.rlhf.dpo_inference as dpo_inference_module

        dpo_inference_module.DPO_MODEL_PATH = self.config.dpo_output_dir
        dpo_inference_module.OUTPUT_DIR = os.path.join(
            get_project_root(),
            "src",
            "semevalpolar",
            "finetuning",
            "rlhf",
            "dpo_predictions",
        )

        print(f"Running inference with model: {self.config.dpo_output_dir}")
        run_dpo_inference(
            input_csv_path=self.config.test_data_path,
            text_column="text",
            output_filename="dpo_predictions.csv",
            max_new_tokens=256,
        )

        predictions_path = os.path.join(
            get_project_root(),
            "src",
            "semevalpolar",
            "finetuning",
            "rlhf",
            "dpo_predictions",
            "dpo_predictions.csv",
        )

        basic_metrics, cm, total_samples, y_true, y_pred = evaluate_dpo_predictions(
            predictions_path
        )
        full_metrics = calculate_all_metrics(y_true, y_pred)
        print_results(full_metrics, cm, total_samples)

        return full_metrics

    def run_all(self, steps=None):
        """
        Run the complete DPO pipeline.

        Args:
            steps: List of steps to run (e.g., [1, 2, 3]). If None, runs all steps.
        """
        if steps is None:
            steps = [1, 2, 3]

        results = {}

        if 1 in steps:
            results["sft"] = self.step1_train_sft()

        if 2 in steps:
            results["dpo"] = self.step2_train_dpo()

        if 3 in steps:
            results["evaluation"] = self.step3_evaluate()

        return results


def main():
    parser = argparse.ArgumentParser(description="DPO Training Pipeline for Qwen Model")
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3],
        help="Run only a specific step (1=SFT, 2=DPO, 3=Evaluate)",
    )
    parser.add_argument(
        "--sft-output", type=str, default=None, help="Custom SFT output directory"
    )
    parser.add_argument(
        "--dpo-output", type=str, default=None, help="Custom DPO output directory"
    )
    parser.add_argument(
        "--sft-adapter-path",
        type=str,
        default=None,
        help="Custom SFT adapter path for DPO training",
    )
    parser.add_argument(
        "--sft-epochs", type=int, default=None, help="SFT number of epochs"
    )
    parser.add_argument("--sft-lr", type=float, default=None, help="SFT learning rate")
    parser.add_argument(
        "--sft-batch-size", type=int, default=None, help="SFT train batch size"
    )
    parser.add_argument(
        "--dpo-epochs", type=int, default=None, help="DPO number of epochs"
    )
    parser.add_argument("--dpo-lr", type=float, default=None, help="DPO learning rate")
    parser.add_argument("--dpo-beta", type=float, default=None, help="DPO beta")
    parser.add_argument(
        "--dpo-batch-size", type=int, default=None, help="DPO train batch size"
    )
    parser.add_argument(
        "--max-length", type=int, default=None, help="Max sequence length"
    )

    args = parser.parse_args()

    config = DPOPipelineConfig()

    if args.sft_output:
        config.sft_output_dir = args.sft_output
    if args.dpo_output:
        config.dpo_output_dir = args.dpo_output
    if args.sft_adapter_path:
        config.sft_adapter_path = args.sft_adapter_path
    if args.sft_epochs is not None:
        config.sft_num_epochs = args.sft_epochs
    if args.sft_lr is not None:
        config.sft_learning_rate = args.sft_lr
    if args.sft_batch_size is not None:
        config.sft_train_batch_size = args.sft_batch_size
    if args.dpo_epochs is not None:
        config.dpo_num_epochs = args.dpo_epochs
    if args.dpo_lr is not None:
        config.dpo_learning_rate = args.dpo_lr
    if args.dpo_beta is not None:
        config.dpo_beta = args.dpo_beta
    if args.dpo_batch_size is not None:
        config.dpo_train_batch_size = args.dpo_batch_size
    if args.max_length is not None:
        config.max_length = args.max_length

    pipeline = DPOPipeline(config)

    if args.step:
        steps = [args.step]
    else:
        steps = [1, 2, 3]

    print(f"Running steps: {steps}")

    if 1 in steps:
        pipeline.step1_train_sft()
    if 2 in steps:
        pipeline.step2_train_dpo()
    if 3 in steps:
        pipeline.step3_evaluate()

    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
