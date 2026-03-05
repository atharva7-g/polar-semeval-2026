import os

import requests
from google import genai

# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# client = genai.Client(api_key=GEMINI_API_KEY)


def get_prompt(statements):
    return f"""
You will receive a list of textual statements. Your task is to classify each
statement as polarizing or non-polarizing.

For EACH statement, output exactly one value:
- "1" if the statement is polarizing
- "0" if the statement is non-polarizing

Definitions and Classification Criteria:

Polarizing statements ("1"):
- Likely to provoke strong disagreement, controversy, or conflict.
- Express or imply advocacy, condemnation, or taking sides on political,
  social, or ideological issues.
- Include calls to action, partisan framing, moral condemnation,
  or emotionally charged language tied to divisive topics.

Non-polarizing statements ("0"):
- Neutral, factual, descriptive, vague, or incomplete.
- May mention controversial people or topics without advocacy or provocation.
- Lack calls to action, strong judgment, or divisive framing.
- Ambiguous or purely observational statements default to non-polarizing.

Important nuances:
- Mere mention of political figures or issues does NOT make a statement polarizing.
- Tone and implied intent matter more than topic presence.
- Strong opinion alone is insufficient unless it is socially divisive.

Input statements:
{statements}

Output format:
Return ONLY a list of strings, one per statement, in the same order.
Each element must be exactly "0" or "1".
No explanations. No extra text.
"""


def get_gepa_prompt(input):
    content = f"""You are given an array of statements. For each statement, classify whether it is polarizing or not polarizing. Output an array of the same length where each element is either "1" (polarizing) or "0" (not polarizing). Do not provide explanations or additional text.

    Input:
    {input}

    Output:
    ["1", "0", ...]
    Return only the raw JSON array (e.g., ["1","0","1"]).
    Do not include code fences, markdown syntax, explanations, or extra text.
    If you are about to include ```json or any other code block formatting, remove it. Output only the plain array.

    Guidelines for classification:
    A statement is polarizing (output "1") if it:
        Uses inflammatory, derogatory, or emotionally charged language (e.g., "dumpster fire democrats," "fake news").
        Attacks, mocks, or vilifies individuals or groups based on political, social, or identity lines.
        Generalizes or stereotypes entire groups (e.g., “MAGA folks don’t have deep nuance to their xenophobia”).
        Uses rhetorical framing intended to provoke division or strong emotional reactions.
    A statement is not polarizing (output "0") if it:
        Reports or describes facts, actions, or opinions using neutral or objective language.
        Mentions political figures, groups, or events without insult or emotional framing.
        Lacks hostility, derision, or attempts to reinforce in-group/out-group division.

    Instructions:

    Only output "1" or "0" for each statement.

    Maintain the original order.

    Do not include punctuation, extra text, or commentary."""

    return content


def build_prompt(input_statements, prompt_path="prompt-ds.txt"):
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    return prompt_template.format(input_statements=input_statements)


def print_response(content):
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=content
    )
    return response.text


def print_local_response(content):
    url = "http://172.24.16.155:11434/api/generate"
    payload = {"model": "llama3.2", "prompt": f"{content}", "stream": False}

    response = requests.post(url, json=payload)
    return response.json()["response"]
