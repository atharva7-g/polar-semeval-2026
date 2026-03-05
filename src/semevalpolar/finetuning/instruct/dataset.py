import torch
from numpy import dtype

from semevalpolar.finetuning.instruct.dataset_utils import extract_label


class PolarDataset(torch.utils.data.Dataset):
    def __init__(self, config, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        polar_label = extract_label(text)
        enc = self.tokenizer(
            text=text,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
            "polar_label": torch.tensor(polar_label, dtype=torch.long),
        }
