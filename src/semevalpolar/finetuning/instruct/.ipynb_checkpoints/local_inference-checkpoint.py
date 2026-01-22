from dataclasses import dataclass
import ollama

@dataclass
class LocalResponseUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float = 0.0


@dataclass
class LocalResponse:
    output_text: str
    usage: LocalResponseUsage


def run_local_ollama(
    example_text: str,
    label: str,
    prompt_path: str,
    model: str,
) -> LocalResponse:
    # load prompt template
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(
        input_statements=example_text,
        ground_truth=label,
    )

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            "temperature": 0.0,
        },
    )

    usage = response.get("usage", {})

    return LocalResponse(
        output_text=response["response"],
        usage=LocalResponseUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            cost=0.0,
        ),
    )
