import torch

class PolarDataset(torch.utils.data.Dataset):
	def __init__(self, config, texts, tokenizer):
		self.texts = texts
		self.tokenizer = tokenizer
		self.config = config

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, idx):
		enc = self.tokenizer(
			text=self.texts[idx],
			truncation=True,
			padding="max_length",
			max_length=self.config.max_length,
			return_tensors="pt"
		)

		input_ids = enc["input_ids"].squeeze(0)
		attention_mask = enc["attention_mask"].squeeze(0)

		return {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"labels": input_ids.clone(),
		}