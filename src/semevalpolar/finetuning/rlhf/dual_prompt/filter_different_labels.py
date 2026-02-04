import json
import re


def extract_label(text):
    """Extract label (0 or 1) from response text."""
    match = re.search(r"Label:\s*(\d+)", text)
    return match.group(1) if match else None


# Read original
with open("preference_pairs.json", "r") as f:
    data = json.load(f)

# Filter pairs where chosen and rejected have DIFFERENT labels
valid_pairs = []
for pair in data["pairs"]:
    chosen_label = extract_label(pair["chosen"])
    rejected_label = extract_label(pair["rejected"])

    # Only keep pairs where labels are different
    if chosen_label and rejected_label and chosen_label != rejected_label:
        valid_pairs.append(pair)

print(f"Original pairs: {len(data['pairs'])}")
print(f"Pairs with DIFFERENT labels: {len(valid_pairs)}")

# Update stats
data["pairs"] = valid_pairs
data["stats"]["total_examples"] = len(valid_pairs)
data["stats"]["examples_with_pairs"] = len(valid_pairs)
data["stats"]["total_pairs_created"] = len(valid_pairs)
data["stats"]["max_pairs_per_example"] = 1

# Save
with open("preference_pairs_cleaned.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved {len(valid_pairs)} pairs to preference_pairs_cleaned.json")
