import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

from semevalpolar.llm.data_utils import read_dataset
from semevalpolar.utils import get_project_root


def generate_message(text, source="tel", target="en"):
	message = [
		{
			"role": "user",
			"content": [
				{
					"type": "text",
					"source_lang_code": source,
					"target_lang_code": target,
					"text": text,
				}
			],
		}
	]

	return message

def translate_texts(
		texts,
		pipe,
		batch_size=8,
		max_new_tokens=200,
):
	translated = []

	for i in tqdm(
			range(0, len(texts), batch_size),
			desc="Translating",
			unit="batch",
	):
		batch = texts[i: i + batch_size]

		messages = [
			generate_message(text)
			for text in batch
		]

		for message in messages:
			output = pipe(
				text=message,
				max_new_tokens=max_new_tokens,
			)

			last_generation = output[0]["generated_text"][-1]
			translated.append(last_generation["content"])

	return translated


# ---------------- Main ----------------
def main():
	# Load dataset
	data_path = os.path.join(
		get_project_root(),
		"data",
		"dev_phase",
		"subtask1",
		"train",
		"tel.csv",
	)

	dataset = read_dataset(data_path)
	texts = dataset["text"]

	# Load translation pipeline
	pipe = pipeline(
		task="image-text-to-text",
		model="google/translategemma-4b-it",
		device="cuda",
		dtype=torch.bfloat16,
	)

	# Translate
	translated_texts = translate_texts(
		texts=texts,
		pipe=pipe,
		batch_size=8,
	)

	# Save results
	dataset["text_en"] = translated_texts
	df = pd.DataFrame(dataset)

	output_path = os.path.join(
		get_project_root(),
		"data",
		"dev_phase",
		"subtask1",
		"train",
		"tel_translated_en.csv",
	)

	df.to_csv(output_path, index=False)
	print(f"Saved translated dataset to: {output_path}")


if __name__ == "__main__":
	main()
