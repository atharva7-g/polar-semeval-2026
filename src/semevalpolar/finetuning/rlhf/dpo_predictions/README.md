# DPO Predictions

This folder contains inference outputs from the DPO-trained model.

## Dataset Used for Training

**Source:** `dual_prompt/dpo_preference_pairs.json`

**Description:** Preference pairs generated using a dual-prompt approach with an LLM. Each pair contains:
- `input`: Raw text to classify
- `chosen`: Assistant response with correct polarization reasoning
- `rejected`: Assistant response with incorrect or weaker polarization reasoning

## Model Details

- **Base Model:** Qwen/Qwen2.5-7B-Instruct
- **Training Method:** DPO (Direct Preference Optimization)
- **Reference:** SFT LoRA checkpoint from `predictions/instruct/final_model`
- **DPO Adapter:** `predictions/instruct/dpo_model_v1`

## Output Files

| File | Description |
|------|-------------|
| `dpo_inference.csv` | Model predictions on input data |

## Pipeline

```
dpo_preference_pairs.json
    ↓
[DPO Training]
    ↓
dpo_model_v1/
    ↓
[Inference]
    ↓
dpo_predictions/dpo_inference.csv
```
