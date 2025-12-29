from datasets import load_dataset
import pandas as pd
import numpy as np

from transformers import (
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import evaluate


def main():
    # Load dataset
    dataset = load_dataset("mteb/tweet_sentiment_extraction")

    # Optional: dataframe view (not used downstream)
    df = pd.DataFrame(dataset["train"])

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Model
    model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2",
        num_labels=3,
    )

    # Small subsets for testing
    small_train_dataset = (
        tokenized_datasets["train"]
        .shuffle(seed=42)
        .select(range(1000))
    )

    small_eval_dataset = (
        tokenized_datasets["test"]
        .shuffle(seed=42)
        .select(range(1000))
    )

    # Metrics
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(
            predictions=predictions,
            references=labels,
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="test_trainer",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate
    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()
