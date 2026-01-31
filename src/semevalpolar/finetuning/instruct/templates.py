import re
from typing import Dict

from semevalpolar.llm.main import create_response


def build_text(x, r, y):
	chat_template = {
		"input": x,
		"reasoning": r,
		"final answer (polarization)": y
	}

	return chat_template

def build_example(x, r, y):
	output = {
		"text": (
			f"Input:\n{x}\n\n"
			f"Reasoning:\n{r}\n\n"
			f"Final Answer:\n{y}"
		)
	}


	return output


def build_prompt(input_statements, ground_truth, prompt_path="prompt-ds.txt"):
	with open(prompt_path, "r", encoding="utf-8") as f:
		prompt_template = f.read()

	return prompt_template.format(input_statements=input_statements, ground_truth={ground_truth})


def run(text, ground_truth, prompt_path="prompt-three-classes.txt", model="qwen/qwen3-max"):
	prompt = build_prompt(text, ground_truth, prompt_path=prompt_path)
	response = create_response(prompt, model=model)

	return response


def parse_prompt(raw: str) -> Dict[str, str]:
	"""
	Split the prompt into sections and collapse any internal newlines
	inside each section to a single space.
	"""
	# Split on headings (case-insensitive, tolerant of extra text)
	pattern = r"(?mi)^(Input|Reasoning|Final Answer(?:.*)?):\s*$"
	parts = re.split(pattern, raw)

	data: Dict[str, str] = {}

	for i in range(1, len(parts) - 1, 2):
		key = parts[i].strip().lower()
		value = parts[i + 1]

		# Normalize whitespace
		value = re.sub(r"[\r\n]+", " ", value)  # remove newlines
		value = re.sub(r"\s+", " ", value)  # collapse spaces
		data[key] = value.strip()

	return data


def parse_prompt_structured(raw: str) -> Dict[str, str]:
	pattern = r"(?mi)^(Input|Reasoning|Final Answer(?:.*)?):\s*$"
	parts = re.split(pattern, raw)

	data: Dict[str, str] = {}

	for i in range(1, len(parts) - 1, 2):
		key = parts[i].strip().lower()
		value = parts[i + 1].strip()
		data[key] = value  # preserve newlines exactly

	return data


def evaluate_metrics(df, y_true_col, y_pred_col):
	y_true = df[y_true_col].values
	y_pred = df[y_pred_col].values

	tp = ((y_true == 1) & (y_pred == 1)).sum()
	tn = ((y_true == 0) & (y_pred == 0)).sum()
	fp = ((y_true == 0) & (y_pred == 1)).sum()
	fn = ((y_true == 1) & (y_pred == 0)).sum()

	accuracy = (tp + tn) / (tp + tn + fp + fn)

	micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	micro_recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	micro_f1 = (
		2 * micro_precision * micro_recall /
		(micro_precision + micro_recall)
		if (micro_precision + micro_recall) > 0
		else 0.0
	)

	# Per-class F1 for macro F1
	# Class 1
	f1_pos = micro_f1

	# Class 0 (treat 0 as positive)
	tp0 = tn
	fp0 = fn
	fn0 = fp

	prec0 = tp0 / (tp0 + fp0) if (tp0 + fp0) > 0 else 0.0
	rec0  = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0.0
	f1_neg = (
		2 * prec0 * rec0 / (prec0 + rec0)
		if (prec0 + rec0) > 0
		else 0.0
	)

	macro_f1 = (f1_pos + f1_neg) / 2

	return {
		"accuracy": accuracy,
		"micro_precision": micro_precision,
		"micro_recall": micro_recall,
		"micro_f1": micro_f1,
		"macro_f1": macro_f1,
	}



def calculate_cost(
	input_tokens: int,
	output_tokens: int,
	input_price_per_million: float = 0.25,
	output_price_per_million: float = 2,
) -> float:
	"""
	Returns total cost in USD for a single response.
	"""
	input_cost = (input_tokens / 1_000_000) * input_price_per_million
	output_cost = (output_tokens / 1_000_000) * output_price_per_million
	return round(input_cost + output_cost, 6)
