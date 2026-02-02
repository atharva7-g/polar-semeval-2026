import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

import re



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

def generate_response(prompt, max_new_tokens=128, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True)

    match = re.search(
        r"Reasoning:\s*[\s\S]*?\n\nFinal label:\s*[01]",
        decoded
    )

    if not match:
        raise ValueError("Invalid model output")

    clean_output = match.group(0)
    return clean_output

if __name__ == "__main__":
    with open("simple-prompt.txt", "r") as f:
        prompt_template = f.read()
    
    input_text = "Donald Trump relies on First Amendment"
    prompt = prompt_template.format(input_text=input_text)

    # For models requiring Llama-style chat format:
    # chat_prompt = f"<s>[INST] {prompt} [/INST]"

    response = generate_response(prompt)
    print(response)

    # Alternative implementation using pipeline:
    # pipe = pipeline(
    #     "text-generation",
    #     model=MODEL_NAME,
    #     torch_dtype=torch.float16,
    #     device_map="auto"
    # )
    # result = pipe(
    #     prompt,
    #     max_new_tokens=256,
    #     temperature=0.7,
    #     do_sample=True,
    #     top_p=0.95,
    #     pad_token_id=tokenizer.eos_token_id
    # )
    # print(result[0]["generated_text"])
