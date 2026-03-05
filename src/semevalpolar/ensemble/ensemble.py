from typing import List

from typing import List
import numpy as np

from semevalpolar.llm.config import data_path
from semevalpolar.llm.data_utils import parse_predictions
from semevalpolar.llm.main import test_run, create_gen
from tqdm import tqdm


def run_base_model(
    prompt: str = "prompts/base-classifier.txt",
    batch_size: int = 10,
    data_path: str = data_path,
    each_prompt_size: int = 2,
    model: str = "qwen/qwen3-max",
    desc: str = "",
) -> List[int]:
    """
    Calls one base LLM with a specific prompt.
    Returns binary labels.
    """

    gen = create_gen(data_path, batch_size=batch_size, randomize=True)
    generator_list = list(gen)

    predictions = []
    usages = []

    for batch in tqdm(generator_list[:each_prompt_size], desc=desc):
        response = test_run(batch, prompt_path=prompt, model=model)
        predictions.append(parse_predictions(response.output_text))
        usages.append(response.usage)

    flat = [x for sub in predictions for x in sub]

    return flat


def proposal_veto_ensemble(preds):
    aggressive_idx = [1, 3]
    conservative_idx = [0]
    veto_idx = [2]

    preds = np.asarray(preds)
    proposal = np.any(preds[aggressive_idx] == 1, axis=0)

    # Any conservative model predicts 1 → veto
    veto = np.any(preds[conservative_idx] == 1, axis=0)

    # Accept proposal unless vetoed
    final = proposal & (~veto)
    return final.astype(int)


def is_polarizing(
    base_model: int,  # 0 or 1
    opinion_filter: int,  # 0 or 1
    rhetoric_gate: int,  # 0 = NO, 1 = YES
    context: int,  # 0 = INSUFFICIENT, 1 = SUFFICIENT
    neutral_style: int,  # 0 = NO, 1 = YES
) -> bool:
    if base_model == 0:
        return False
    if opinion_filter == 0:
        return False
    if rhetoric_gate == 0:  # NO
        return False
    if context == 0:  # INSUFFICIENT
        return False
    if neutral_style == 1:  # YES
        return False
    return True
