import inspect
import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import evaluate
import numpy as np
import torch
import yaml
from datasets import Dataset, DatasetDict
from torch import nn
from transformers import (
	AutoTokenizer,
	AutoModelForSequenceClassification,
	TrainingArguments,
	Trainer,
)

# Assuming these exist in your project structure
from semevalpolar.llm.data_utils import read_dataset, split_dataframe
from semevalpolar.utils import get_project_root


@dataclass(frozen=True)
class TrainingConfig:
	model_name: str = "distilbert-base-cased"
	num_labels: int = 2
	max_length: int = 512
	# Use default_factory to calculate path at runtime, preventing import errors
	output_dir: str = field(default_factory=lambda: os.path.join(get_project_root(), "predictions", "finetuning"))
	eval_strategy: str = "epoch"
	train_batch_size: int = 4
	eval_batch_size: int = 4
	data_path: str = field(default_factory=lambda: os.path.join(get_project_root(), "relabelling", "eng.csv"))
	num_train_epochs: int = 3
	learning_rate: float = 5e-5

	pretrained_path: Optional[str] = None

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


def _resolve_model_checkpoint(config: TrainingConfig) -> str:
	if config.pretrained_path:
		cand = os.path.abspath(config.pretrained_path)
		# a fine‑tuned checkpoint must contain at least a config.json
		if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "config.json")):
			return cand
		else:
			print(
				f"pretrained_path '{cand}' does not look like a HF checkpoint – "
				"falling back to model_name."
			)
	return config.model_name


def load_config(path: str = None) -> TrainingConfig:
	# Set default path here to ensure get_project_root() is called at runtime
	if path is None:
		path = os.path.join(get_project_root(), "src", "semevalpolar", "finetuning", "slm", "config", "config.yaml")
	return TrainingConfig.from_yaml(path)


class PolarizationDatasetBuilder:
	def __init__(self, tokenizer, max_length: int):
		self.tokenizer = tokenizer
		self.max_length = max_length

	@staticmethod
	def _prepare_dataframe(df):
		# Ensure column exists before renaming
		if "polarization" in df.columns:
			df = df.rename(columns={"polarization": "label"})

		# Ensure label is int type for classification
		if "label" in df.columns:
			df["label"] = df["label"].astype(int)
		else:
			raise ValueError("Dataframe must contain 'polarization' or 'label' column.")

		# HuggingFace datasets don't handle complex indices well
		df = df.reset_index(drop=True)
		return df

	def _tokenize(self, examples):
		return self.tokenizer(
			examples["text"],
			padding="max_length",
			truncation=True,
			max_length=self.max_length,
		)

	def build(self, csv_path: str) -> DatasetDict:
		if not os.path.exists(csv_path):
			raise FileNotFoundError(f"Dataset not found at {csv_path}")

		raw_df = read_dataset(csv_path)
		train_df, val_df, test_df = split_dataframe(raw_df, random_state=40)

		train_df = self._prepare_dataframe(train_df)
		val_df = self._prepare_dataframe(val_df)
		test_df = self._prepare_dataframe(test_df)

		dataset = DatasetDict(
			{
				"train": Dataset.from_pandas(train_df),
				"validation": Dataset.from_pandas(val_df),
				"test": Dataset.from_pandas(test_df),
			}
		)

		# Remove extra index columns if they appeared during conversion
		cols_to_remove = [c for c in dataset["train"].column_names if c.startswith("__index")]
		if cols_to_remove:
			dataset = dataset.remove_columns(cols_to_remove)

		return dataset.map(self._tokenize, batched=True)


class PrecisionMetric:
	def __init__(self, average: str = "binary"):
		self.metric = evaluate.load("precision")
		self.average = average

	def __call__(self, eval_pred):
		logits, labels = eval_pred
		predictions = np.argmax(logits, axis=-1)

		return self.metric.compute(predictions=predictions, references=labels, average=self.average)


class AccuracyMetric:
	def __init__(self):
		self.metric = evaluate.load("accuracy")

	def __call__(self, eval_pred):
		logits, labels = eval_pred
		predictions = np.argmax(logits, axis=-1)
		return self.metric.compute(predictions=predictions, references=labels)


class Metrics:
	def __init__(self, average: str = "binary"):
		self.accuracy = evaluate.load("accuracy")
		self.precision = evaluate.load("precision")
		self.f1 = evaluate.load("f1")
		self.average = average

	def __call__(self, eval_pred):
		logits, labels = eval_pred
		predictions = np.argmax(logits, axis=-1)

		# Calculate all metrics
		acc_score = self.accuracy.compute(predictions=predictions, references=labels)
		prec_score = self.precision.compute(predictions=predictions, references=labels, average=self.average)
		f1_score = self.f1.compute(predictions=predictions, references=labels, average=self.average)

		return {**acc_score, **prec_score, **f1_score}


class WeightedTrainer(Trainer):
	def compute_loss(
			self,
			model: nn.Module,
			inputs: dict[str, Union[torch.Tensor, Any]],
			return_outputs: bool = False,
			num_items_in_batch: Optional[torch.Tensor] = None,
	):
		labels = inputs.get("labels")
		outputs = model(**inputs)
		logits = outputs.get("logits")

		loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 2.5]))
		loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

		return (loss, outputs) if return_outputs else loss


class TrainingPipeline:
	def __init__(self, config: TrainingConfig, tokenizer):
		self.config = config
		self.tokenizer = tokenizer

		model_checkpoint = _resolve_model_checkpoint(config)
		self.model = AutoModelForSequenceClassification.from_pretrained(
			model_checkpoint,
			num_labels=config.num_labels,
		)

		# IMPORTANT: Resize must happen if you added tokens (like a new PAD token)
		# Check if tokenizer size matches model size; if not, resize
		if len(tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
			print("Resizing model embeddings to match tokenizer...")
			self.model.resize_token_embeddings(len(tokenizer))

		# Explicitly set the pad token id in the model config
		self.model.config.pad_token_id = tokenizer.pad_token_id
		self.metric = Metrics(average="macro")

	def _build_training_args(self) -> TrainingArguments:
		return TrainingArguments(
			output_dir=self.config.output_dir,
			eval_strategy=self.config.eval_strategy,
			push_to_hub=False,
			per_device_train_batch_size=self.config.train_batch_size,
			per_device_eval_batch_size=self.config.eval_batch_size,
			logging_strategy="epoch",
			save_strategy="epoch",
			load_best_model_at_end=True,  # Recommended: load best model after training
			metric_for_best_model="precision",
			num_train_epochs=self.config.num_train_epochs,
			learning_rate=self.config.learning_rate,
		)

	def run(self, dataset: DatasetDict):
		trainer = WeightedTrainer(
			model=self.model,
			args=self._build_training_args(),
			train_dataset=dataset["train"],
			# Use validation set for training evaluation
			eval_dataset=dataset["validation"],
			compute_metrics=self.metric,
			tokenizer=self.tokenizer,
		)

		print("Starting training...")
		trainer.train()

		# Run final evaluation on the held-out TEST set
		print("Running final evaluation on test set...")
		test_results = trainer.evaluate(dataset["test"])
		print(f"Test Set Results: {test_results}")

		if not self.config.pretrained_path:
			save_path = os.path.join(self.config.output_dir, "final_model")
		else:
			save_path = os.path.join(self.config.output_dir, "final_finetuned_model")
		self.model.save_pretrained(save_path)
		self.tokenizer.save_pretrained(save_path)
		print(f"Model and tokenizer saved to {save_path}")


def main():
	config = load_config()

	print(f"Loading tokenizer: {config.model_name}")
	tokenizer = AutoTokenizer.from_pretrained(config.model_name)

	# FIX 1: DistilBERT and BERT are encoder models and generally expect RIGHT padding.
	# Left padding is usually for generation (decoder) models.
	tokenizer.padding_side = "right"

	# FIX 2: Handle Pad Token intelligently.
	# If the tokenizer doesn't have a pad token, add one.
	if tokenizer.pad_token is None:
		print("Adding [PAD] token...")
		tokenizer.add_special_tokens({'pad_token': '[PAD]'})
		# Note: We must resize embeddings in the pipeline after loading the model

	# Build dataset
	dataset_builder = PolarizationDatasetBuilder(
		tokenizer=tokenizer,
		max_length=config.max_length,
	)

	dataset = dataset_builder.build(config.data_path)

	# Run training
	pipeline = TrainingPipeline(config, tokenizer)
	pipeline.run(dataset)


if __name__ == "__main__":
	main()
