import inspect
import os
from dataclasses import dataclass, field
from typing import Union, Any, Optional, Dict
import torch
from peft import LoraConfig, get_peft_model, TaskType

import yaml
from datasets import load_dataset
from torch import nn
from transformers import AutoModelForCausalLM, Trainer, AutoTokenizer
from transformers import TrainingArguments

from semevalpolar.finetuning.instruct.dataset import PolarDataset
from semevalpolar.utils import get_project_root


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_length: int = 1024
    train_batch_size: int = 1
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 10
    learning_rate: float = 5e-5
    train_data_path: str = field(
        default_factory=lambda: os.path.join(
            get_project_root(),
            "src",
            "semevalpolar",
            "finetuning",
            "instruct",
            "data",
            "dataset.jsonl",
        )
    )
    output_dir: str = field(
        default_factory=lambda: os.path.join(
            get_project_root(), "predictions", "instruct"
        )
    )
    dtype: str = "fp16"

    @classmethod
    def from_yaml(cls, path: str):
        if not os.path.exists(path):
            print(f"Config file not found at {path}. Using defaults.")
            return cls()

        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)

        if cfg_dict is None:
            return cls()

        # Filter out keys in YAML that are not in the dataclass to prevent crashes
        valid_keys = inspect.signature(cls).parameters.keys()
        filtered_dict = {k: v for k, v in cfg_dict.items() if k in valid_keys}

        if "learning_rate" in filtered_dict:
            filtered_dict["learning_rate"] = float(filtered_dict["learning_rate"])

        return cls(**filtered_dict)


def load_config(path: str = None) -> TrainingConfig:
    if path is None:
        path = os.path.join(
            get_project_root(),
            "src",
            "semevalpolar",
            "finetuning",
            "instruct",
            "config",
            "config.yaml",
        )
    return TrainingConfig.from_yaml(path)


class WeightedTrainer(Trainer):
    def __init__(self, *, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
        **kwargs,
    ):
        polar_labels = inputs.pop("polar_label")
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        ).view(shift_labels.size())

        valid_tokens = (shift_labels != -100).float()
        sample_loss = (token_loss * valid_tokens).sum(dim=1) / (
            valid_tokens.sum(dim=1) + 1e-8
        )

        if self.class_weights.device != sample_loss.device:
            self.class_weights = self.class_weights.to(sample_loss.device)

        batch_weights = self.class_weights[polar_labels]
        loss = (sample_loss * batch_weights).mean()

        return (loss, outputs) if return_outputs else loss


class TrainingPipeline:
    def __init__(self, config: TrainingConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=torch.float16,
        )

        current_embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) != current_embedding_size:
            print(
                f"Resizing embeddings from {current_embedding_size} to {len(self.tokenizer)}"
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            bias="none",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.enable_input_require_grads()
        self.model.print_trainable_parameters()

        self.model.config.pad_token_id = tokenizer.pad_token_id
        self.model.config.bos_token_id = tokenizer.bos_token_id
        self.model.config.eos_token_id = tokenizer.eos_token_id

        self.model.generation_config.pad_token_id = tokenizer.pad_token_id
        self.model.generation_config.bos_token_id = tokenizer.bos_token_id
        self.model.generation_config.eos_token_id = tokenizer.eos_token_id

        self.model.config.pad_token_id = tokenizer.pad_token_id

    def _build_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            push_to_hub=False,
            logging_steps=50,
            save_steps=1000,
            save_total_limit=2,
            eval_strategy="no",
            optim="adamw_torch",
            fp16=True,
            bf16=False,
            remove_unused_columns=False,
            gradient_checkpointing=False,
            dataloader_num_workers=0,
        )

    def run(self, train_dataset):
        trainer = WeightedTrainer(
            class_weights=[1.0, 1.5],
            model=self.model,
            args=self._build_training_args(),
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
        )
        print("Model device:", next(self.model.parameters()).device)
        print("Starting training...")
        trainer.train()

        save_path = os.path.join(self.config.output_dir, "sft_model")

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")

    print("Loading config...")
    config = load_config()

    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading dataset...")
    train_data = load_dataset("json", data_files=config.train_data_path)["train"]

    texts = train_data["text"]

    print("Building dataset...")
    train_dataset = PolarDataset(
        texts=texts,
        config=config,
        tokenizer=tokenizer,
    )

    sample = train_dataset[0]
    assert "polar_label" in sample

    print("Starting training pipeline...")
    pipeline = TrainingPipeline(
        config=config,
        tokenizer=tokenizer,
    )

    pipeline.run(train_dataset)


if __name__ == "__main__":
    main()
