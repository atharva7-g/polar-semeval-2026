import json
from openai import OpenAI
from tqdm import tqdm
import os
from semevalpolar.utils import get_project_root

import time

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

INPUT_FILE = (
    f"{get_project_root()}/src/semevalpolar/finetuning/rlhf/preference_pairs_v8.json"
)
OUTPUT_FILE = (
    f"{get_project_root()}/src/semevalpolar/finetuning/rlhf/clean_pairs_v8.json"
)

BATCH_SIZE = 16


with open(INPUT_FILE) as f:
    data = json.load(f)

pairs = data["pairs"]
clean_pairs = []


def build_batch_prompt(batch):
    formatted = []

    for i, pair in enumerate(batch):
        formatted.append(f"""
Pair {i}

Prompt:
{pair["prompt"]}

Chosen:
{pair["chosen"]}

Rejected:
{pair["rejected"]}
""")

    pairs_text = "\n".join(formatted)

    return f"""
You are validating RLHF preference pairs.

For each pair determine:
1. Is the prompt ambiguous?
2. Is the chosen response clearly better?

Return exactly one word per pair.

VALID
or
INVALID

Output format:
Pair 0: VALID
Pair 1: INVALID
...

Pairs:

{pairs_text}
"""


for i in tqdm(range(0, len(pairs), BATCH_SIZE)):
    batch = pairs[i : i + BATCH_SIZE]

    prompt = build_batch_prompt(batch)

    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    output = response.choices[0].message.content

    decisions = {}

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Pair"):
            idx, decision = line.split(":")
            idx = int(idx.replace("Pair", "").strip())
            decision = decision.strip()
            decisions[idx] = decision

    for j, pair in enumerate(batch):
        if decisions.get(j) == "VALID":
            clean_pairs.append(pair)

print("Original pairs:", len(pairs))
print("Clean pairs:", len(clean_pairs))

data["pairs"] = clean_pairs

with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=2)
