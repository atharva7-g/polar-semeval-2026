import pandas as pd
import re
from typing import Callable, Optional, List


# -----------------------------
# Rule function type
# -----------------------------
RuleFn = Callable[[str, int], Optional[int]]
# takes (text, current_label) → returns new_label or None



# -----------------------------
# Helper matchers
# -----------------------------
def contains_any(text: str, keywords: List[str]) -> bool:
    text_l = text.lower()
    return any(k in text_l for k in keywords)


def looks_like_metadata(text: str) -> bool:
    return bool(re.fullmatch(r"(source\s+cnn|cnn\s+business|\s*)+", text.lower()))


def is_truncated(text: str) -> bool:
    return text.strip().endswith(",")


# -----------------------------
# Example rule functions
# -----------------------------
def rule_force_non_polarizing_metadata(text: str, label: int) -> Optional[int]:
    if looks_like_metadata(text):
        return 0
    return None


def rule_force_non_polarizing_truncated(text: str, label: int) -> Optional[int]:
    if is_truncated(text):
        return 0
    return None


def rule_force_polarizing_veterans(text: str, label: int) -> Optional[int]:
    if contains_any(text, ["honors veterans", "salute to service"]):
        return 1
    return None


def rule_force_polarizing_refugees(text: str, label: int) -> Optional[int]:
    if contains_any(text, ["refugee", "refugees", "acnur", "unhcr"]):
        return 1
    return None

def rule_institutional_document(text, label):
    if label == 1:
        if (
            len(text.split()) > 6
            and text == text.title()
            and not re.search(r"[!?]", text)
        ):
            return 0
    return None

def rule_low_entropy(text, label):
    tokens = text.lower().split()
    if len(set(tokens)) <= 3 and len(tokens) > 4:
        return 0
    return None

def rule_no_explicit_target(text, label):
    if label == 1:
        if not re.search(r"\b(democrats|republicans|government|state|media|israel|russia|china)\b", text.lower()):
            return 0
    return None



rules_default = [
    rule_force_non_polarizing_metadata,
    rule_force_non_polarizing_truncated,
    rule_force_polarizing_veterans,
    rule_force_polarizing_refugees,
    rule_institutional_document,
    rule_low_entropy,
    rule_no_explicit_target
]

# -----------------------------
# Rule engine
# -----------------------------
def apply_rules(
    df: pd.DataFrame,
    text_col: str = "Text",
    pred_col: str = "Predicted",
    rules: List[RuleFn] = None,
) -> pd.DataFrame:
    if rules is None:
        rules = rules_default

    def apply_row(row):
        label = row[pred_col]
        text = row[text_col]

        for rule in rules:
            new_label = rule(text, label)
            if new_label is not None:
                label = new_label

        return label

    df = df.copy()
    df["final_label"] = df.apply(apply_row, axis=1)
    return df
