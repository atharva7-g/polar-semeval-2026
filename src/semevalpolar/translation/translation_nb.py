import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    from transformers import pipeline
    import torch
    return pipeline, torch


@app.cell
def _(pipeline, torch):
    pipe = pipeline(
        "image-text-to-text",
        model="google/translategemma-4b-it",
        device="cuda",
        dtype=torch.bfloat16
    )
    return (pipe,)


@app.cell
def _():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": "en",
                    "target_lang_code": "hi",
                    "text": "How are you?",
                }
            ],
        }
    ]

    return (messages,)


@app.cell
def _(messages, pipe):
    output = pipe(text=messages, max_new_tokens=200)
    return (output,)


@app.cell
def _(output):
    print(output[0]["generated_text"][-1]["content"])
    return


if __name__ == "__main__":
    app.run()
