import json


def clean_response(text):
    """Remove 'Statement:\n...' block, keep only Explanation and Label"""
    if text.startswith("Statement:\n"):
        explanation_start = text.find("\n\nExplanation:")
        if explanation_start != -1:
            text = text[explanation_start:]
    # Strip leading newlines
    return text.lstrip("\n")


def is_valid_response(text):
    """Check if response follows proper 'Explanation:\n...' format (not template text)"""
    # Must start with "Explanation:" and not contain "Okay, I understand"
    return text.startswith("Explanation:") and "Okay, I understand" not in text


# Read original
with open("preference_pairs.json", "r") as f:
    data = json.load(f)

original_count = len(data["pairs"])

# Filter pairs with valid responses
valid_pairs = []
for pair in data["pairs"]:
    # Clean responses first
    cleaned_chosen = clean_response(pair["chosen"])
    cleaned_rejected = clean_response(pair["rejected"])

    # Check if cleaned responses are valid (start with "Explanation:" and no template text)
    if (
        cleaned_chosen.startswith("Explanation:")
        and "Okay, I understand" not in cleaned_chosen
    ):
        if (
            cleaned_rejected.startswith("Explanation:")
            and "Okay, I understand" not in cleaned_rejected
        ):
            pair["chosen"] = cleaned_chosen
            pair["rejected"] = cleaned_rejected
            valid_pairs.append(pair)

# Update data with valid pairs
data["pairs"] = valid_pairs
data["stats"]["total_examples"] = len(valid_pairs)
data["stats"]["examples_with_pairs"] = len(valid_pairs)
data["stats"]["total_pairs_created"] = len(valid_pairs)
data["stats"]["max_pairs_per_example"] = 1

# Save cleaned version
with open("preference_pairs_cleaned.json", "w") as f:
    json.dump(data, f, indent=2)

print(
    f"Filtered to {len(valid_pairs)} valid pairs (removed {original_count - len(valid_pairs)} invalid entries)"
)
print("Output saved to preference_pairs_cleaned.json")
