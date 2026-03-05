import os
import pandas as pd
from random import randint

from semevalpolar.utils import get_project_root
from semevalpolar.llm.data_utils import read_dataset, create_submission
from semevalpolar.finetuning.slm.inference import run_inference_on_df


def main():
    project_root = get_project_root()

    # Load dataset
    df = read_dataset(
        os.path.join(project_root, "data", "test_phase", "subtask1", "test", "urd.csv")
    )

    # Set model checkpoint directory
    model_dir = os.path.join(
        project_root, "src", "predictions", "finetuning", "final_model"
    )

    # Run inference on full dataset
    predictions = run_inference_on_df(
        df=df, text_column="text", model_dir=model_dir, batch_size=16
    )

    # Create and save submission
    submission = create_submission(df, predictions)
    submission_path = os.path.join(
        project_root, "predictions", "subtask_1", "pred_urd.csv"
    )
    submission.to_csv(submission_path, index=False)


if __name__ == "__main__":
    main()
