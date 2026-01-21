from transformers import pipeline
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/translategemma-27b-it",
    device="cuda",
    dtype=torch.bfloat16
)

# ---- Text Translation ----
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "source_lang_code": "cs",
                "target_lang_code": "de-DE",
                "text": "V nejhorším případě i k prasknutí čočky.",
            }
        ],
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])