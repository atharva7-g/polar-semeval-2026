import inspect
import os
from dataclasses import dataclass, field

import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, AutoTokenizer
from transformers import TrainingArguments

from semevalpolar.finetuning.instruct.dataset import PolarDataset
from semevalpolar.utils import get_project_root


@dataclass(frozen=True)
class TrainingConfig:
	model_name: str = "Qwen/Qwen2.5-7B-Instruct"
	max_length: int = 2048
	train_batch_size: int = 4
	eval_batch_size: int = 4
	gradient_accumulation_steps: int = 16
	num_train_epochs: int = 3
	learning_rate: float = 5e-5
	train_data_path: str = field(default_factory=lambda: os.path.join(get_project_root(),
	                                                            "src",
	                                                            "semevalpolar", "finetuning",
	                                                            "instruct",
	                                                            "data", "splits", "train.jsonl"))
	output_dir: str = field(default_factory=lambda: os.path.join(get_project_root(), "predictions", "instruct"))

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
		path = os.path.join(get_project_root(), "src", "semevalpolar", "finetuning",
		                    "instruct", "config", "config.yaml")
	return TrainingConfig.from_yaml(path)


class TrainingPipeline:
	def __init__(self, config: TrainingConfig, tokenizer):
		self.config = config
		self.tokenizer = tokenizer

		self.model = AutoModelForCausalLM.from_pretrained(
			self.config.model_name,
			torch_dtype="bfloat16",
			device_map="auto"
		)

		if len(tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
			print("Resizing model embeddings to match tokenizer...")
			self.model.resize_token_embeddings(len(tokenizer))

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
			bf16=True,
		)

	def run(self, train_dataset):
		trainer = Trainer(
			model=self.model,
			args=self._build_training_args(),
			train_dataset=train_dataset,
			tokenizer=self.tokenizer,
		)

		print("Starting training...")
		trainer.train()

		save_path = os.path.join(self.config.output_dir, "final_model")

		self.model.save_pretrained(save_path)
		self.tokenizer.save_pretrained(save_path)
		print(f"Model and tokenizer saved to {save_path}")


def main():
	print("Loading config...")
	config = load_config()

	print(f"Loading tokenizer: {config.model_name}")
	tokenizer = AutoTokenizer.from_pretrained(config.model_name)

	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "right"

	train_data = load_dataset("json", data_files=config.train_data_path)["train"]
	texts = train_data["text"]

	train_dataset = PolarDataset(
		texts=texts,
		config=config,
		tokenizer=tokenizer,
	)

	pipeline = TrainingPipeline(config=config, tokenizer=tokenizer)
	pipeline.run(train_dataset)


if __name__ == "__main__":
	main()
