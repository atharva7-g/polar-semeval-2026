from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_MODEL_CACHE = {}

@dataclass
class LocalResponseUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float = 0.0  # local = free 🙂


@dataclass
class LocalResponse:
    output_text: str
    usage: LocalResponseUsage


def get_hf_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        model.eval()
        _MODEL_CACHE[model_name] = (tokenizer, model)

    return _MODEL_CACHE[model_name]

def run_local_hf(
    example_text: str,
    label: str,
    prompt_path: str,
    model_name: str,
    max_new_tokens: int = 512,
) -> LocalResponse:
    tokenizer, model = get_hf_model(model_name)

    # load prompt template
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(
        text=example_text,
        label=label,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_tokens = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    output_tokens = outputs.shape[1] - input_tokens
    total_tokens = outputs.shape[1]

    decoded = tokenizer.decode(
        outputs[0][input_tokens:],
        skip_special_tokens=True,
    )

    return LocalResponse(
        output_text=decoded,
        usage=LocalResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=0.0,
        ),
    )

