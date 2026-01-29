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
