from tqdm import tqdm
from typing import List, Tuple, Any


def run_examples_with_tqdm(
    example_df,
    run_fn,
    parse_fn,
    build_text_fn,
    prompt_path: str,
    model: str,
    limit: int | None = None,
    desc: str = "Running examples",
) -> dict:
    dataset: List[Any] = []
    responses: List[Any] = []

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_cost = 0.0

    df = example_df.head(limit) if limit else example_df

    pbar = tqdm(
        df.iterrows(),
        total=len(df),
        desc=desc,
    )

    for _, row in pbar:
        example_text = row["text"]
        label = row["ground_truth"]

        response = run_fn(
            example_text,
            label,
            prompt_path=prompt_path,
            model=model,
        )

        parsed = parse_fn(response.output_text)
        dataset.append(
            build_text_fn(
                parsed["input"],
                parsed["reasoning"],
                parsed["final answer"],
            )
        )
        responses.append(parsed)

        # --- usage tracking ---
        usage = getattr(response, "usage", None)
        if usage:
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
            total_tokens += usage.total_tokens
            total_cost += usage.cost

            pbar.set_postfix(
                in_tok=total_input_tokens,
                out_tok=total_output_tokens,
                total_tok=total_tokens,
                cost=f"${total_cost:.6f}",
            )

    return {
        "dataset": dataset,
        "responses": responses,
        "totals": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": total_cost,
        },
    }
