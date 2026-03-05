import pandas as pd
from sklearn.model_selection import train_test_split
import ast
import numpy as np


def read_dataset(filename):
    data = pd.read_csv(filename)
    return data


def split_dataframe(
    df: pd.DataFrame,
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    random_state=42,
    shuffle=True,
):
    if not abs(train_size + val_size + test_size - 1.0) < 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    train_df, temp_df = train_test_split(
        df, train_size=train_size, random_state=random_state, shuffle=shuffle
    )
    relative_val_size = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    return train_df, val_df, test_df


def batch_df(df, batch_size=100, randomize=True, random_state=42):
    if randomize:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(len(df))
        df = df.iloc[indices]
    for i in range(0, len(df), batch_size):
        yield df.iloc[i : i + batch_size]


def parse_predictions(response):
    try:
        predicted = ast.literal_eval(response)
        if not isinstance(predicted, list):
            raise ValueError("Parsed output is not a list.")

        return [1 if str(x).strip() in {"Yes", "1"} else 0 for x in predicted]

    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid response format: {e}")


def create_submission(df, predictions):
    submission_df = pd.DataFrame({"id": df["id"], "polarization": predictions})

    return submission_df


def create_comparison_df(predicted, ground_truth, text):
    min_len = min(len(predicted), len(ground_truth))
    df = pd.DataFrame(
        {
            "Predicted": predicted[:min_len],
            "Ground Truth": ground_truth[:min_len],
            "Text": text[:min_len],
        }
    )
    return df
