import os
from dataclasses import dataclass

import numpy as np
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import yaml

from semevalpolar.llm.data_utils import read_dataset, split_dataframe
from semevalpolar.utils import get_project_root


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "distilbert-base-cased"
    num_labels: int = 2
    max_length: int = 512
    output_dir: str = os.path.join(get_project_root(), "predictions", "finetuning")
    eval_strategy: str = "epoch"
    train_batch_size: int = 4
    eval_batch_size: int = 4

def load_config(path: str = "config.yaml") -> TrainingConfig:
    if os.path.exists(path):
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        return TrainingConfig(**{**TrainingConfig().__dict__, **cfg_dict})
    else:
        return TrainingConfig()


class PolarizationDatasetBuilder:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def _prepare_dataframe(df):
        df = df.rename(columns={"polarization": "label"})
        df["label"] = df["label"].astype(int)
        return df

    def _tokenize(self, examples):
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

    def build(self, csv_path: str) -> DatasetDict:
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

        if "__index_level_0__" in dataset["train"].column_names:
            dataset = dataset.remove_columns("__index_level_0__")

        return dataset.map(self._tokenize, batched=True)


class AccuracyMetric:
    def __init__(self):
        self.metric = evaluate.load("accuracy")

    def __call__(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)


class TrainingPipeline:
    def __init__(self, config: TrainingConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        # Load model and resize embeddings for PAD token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
        )
        self.model.resize_token_embeddings(len(tokenizer))

        self.model.config.pad_token_id = tokenizer.pad_token_id

        self.metric = AccuracyMetric()

    def _build_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.config.output_dir,
            eval_strategy=self.config.eval_strategy,
            push_to_hub=False,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            logging_strategy="epoch",
            save_strategy="epoch",
        )

    def run(self, dataset: DatasetDict):
        trainer = Trainer(
            model=self.model,
            args=self._build_training_args(),
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=self.metric,
            tokenizer=self.tokenizer,
        )
        trainer.train()

        save_path = os.path.join(self.config.output_dir, "saved_model")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")


def main():
    config = TrainingConfig()

    # Load tokenizer and add PAD token if missing
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    # Build dataset
    dataset_builder = PolarizationDatasetBuilder(
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    data_path = os.path.join(
        get_project_root(),
        "data",
        "relabelling",
        "eng.csv",
    )

    dataset = dataset_builder.build(data_path)

    # Run training
    pipeline = TrainingPipeline(config, tokenizer)
    pipeline.run(dataset)


if __name__ == "__main__":
    main()
