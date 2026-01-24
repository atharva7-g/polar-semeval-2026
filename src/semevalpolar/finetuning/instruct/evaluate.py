import os
import re
from typing import List

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from semevalpolar.finetuning.instruct.finetune import load_config
from semevalpolar.utils import get_project_root


def predict_polarization(prompts: List[str]) -> List[int]:
	"""
	Runs POLAR inference on a list of raw input strings.

	Args:
		prompts: list of input texts (NOT full prompts)

	Returns:
		List of predicted labels (0 or 1), one per input.
	"""

	def extract_label(text: str):
		match = re.search(r"Final Answer:\s*([01])", text)
		return int(match.group(1)) if match else None

	config = load_config()
	adapter_path = os.path.join(
		get_project_root(),
		"predictions",
		"instruct",
		"final_model"
	)

	tokenizer = AutoTokenizer.from_pretrained(adapter_path)

	base_model = AutoModelForCausalLM.from_pretrained(
		config.model_name,
		torch_dtype=torch.bfloat16,
		device_map="auto"
	)
	base_model.resize_token_embeddings(len(tokenizer))

	model = PeftModel.from_pretrained(base_model, adapter_path)
	model.eval()

	predictions = []

	for text in prompts:
		prompt = f"""Input:
				{text}

				Reasoning:
				"""

		inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

		outputs = model.generate(
			**inputs,
			max_new_tokens=256,
			do_sample=False,
			eos_token_id=tokenizer.eos_token_id
		)

		decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
		label = extract_label(decoded)

		if label is None:
			raise RuntimeError(f"Failed to extract label from output:\n{decoded}")

		predictions.append(label)

	return predictions
