# MATRIX Evaluation Artifact

Minimal inference and evaluation code for the anonymized MATRIX dataset artifact.

Dataset:

```text
https://huggingface.co/datasets/luckyorangepepper/anon-pepper-72
```

Judge:

```text
claude-opus-4-5
```

- `inference/`: general model inference for text Q&A and image-caption rows.
- `evaluation/`: Opus judge scoring and fixed rubrics.
- `finetune/`: Qwen and Llama-3.2-11B-Vision post-training scripts.
- `cka/`: Qwen-only image-conditioned CKA analysis used in the paper.
- `examples/`: minimal accepted prediction schemas.

## Install

```bash
pip install -r requirements.txt
```

## Export Finetune Data

```bash
bash scripts/prepare_matrix_data.sh
```

This writes `train`, `validation`, and `test` JSONL files under
`finetune/exports/` and materializes image files under
`finetune/exports/images/`.

## Post-Train Models

Qwen MATRIX-QPT runs:

```bash
MODE=text RUN_NAME=qwen-text-sft bash scripts/finetune_qwen.sh
MODE=vision RUN_NAME=qwen-vision-sft bash scripts/finetune_qwen.sh
MODE=full RUN_NAME=qwen-full-sft bash scripts/finetune_qwen.sh
```

Llama MATRIX-LPT runs:

```bash
MODE=text RUN_NAME=llama32-text-sft bash scripts/finetune_llama32_11b.sh
MODE=vision RUN_NAME=llama32-vision-sft bash scripts/finetune_llama32_11b.sh
MODE=full RUN_NAME=llama32-full-sft bash scripts/finetune_llama32_11b.sh
```

Llama-3.2-11B-Vision is gated on Hugging Face, so run `huggingface-cli login`
with an account that has accepted the model license before launching Llama
post-training.

## Reproduce CKA

The representational analysis in the paper is Qwen-only. It uses the base
Qwen2-VL-7B model and the Qwen text, vision, and full adapters.

```bash
bash scripts/reproduce_cka.sh
```

## Generate Predictions

The evaluator scores model outputs; it does not call the model itself. Use
`inference/generate_predictions.py` to produce the prediction JSONL first. The
script accepts a Hugging Face model ID, a local full-model checkpoint, or a LoRA
adapter directory with `adapter_config.json`.

Text Q&A / hypothesis rows:

```bash
python inference/generate_predictions.py \
  --model-path mistralai/Ministral-3-8B-Instruct-2512 \
  --dataset luckyorangepepper/anon-pepper-72 \
  --split test \
  --task text \
  --output-file outputs/text_predictions.jsonl \
  --max-new-tokens 256
```

Image-caption rows:

```bash
python inference/generate_predictions.py \
  --model-path Qwen/Qwen2-VL-7B-Instruct \
  --dataset luckyorangepepper/anon-pepper-72 \
  --split test \
  --task caption \
  --output-file outputs/image_predictions.jsonl \
  --max-new-tokens 128
```

For exported/local JSONL files with relative image paths, add `--input-file` and
`--image-root`:

```bash
python inference/generate_predictions.py \
  --model-path /path/to/adapter-or-model \
  --base-model Qwen/Qwen2-VL-7B \
  --input-file finetune/exports/test.jsonl \
  --image-root finetune/exports/images/test \
  --task caption \
  --output-file outputs/local_image_predictions.jsonl
```

Run inference and Opus scoring in one command:

```bash
ANTHROPIC_API_KEY=... python run_and_score.py \
  --model-path Qwen/Qwen2-VL-7B-Instruct \
  --dataset luckyorangepepper/anon-pepper-72 \
  --split test \
  --task auto \
  --predictions-file outputs/model_predictions.jsonl \
  --scores-file outputs/model_scores.jsonl \
  --summary-csv outputs/model_summary.csv
```

## Score Text Outputs

Prepare a JSONL file with one row per example:

```json
{"qid":"example-1","type":"text","kind":"theory","question":"...","answer":"...","model_answer":"..."}
```

See `examples/text_predictions.jsonl` for a minimal text row.

If the prediction file already includes `question`, `answer`, and
`model_answer`, score it directly:

```bash
ANTHROPIC_API_KEY=... python evaluate.py \
  --answers-file outputs/model_predictions.jsonl \
  --output-file outputs/model_scores.jsonl \
  --summary-csv outputs/model_summary.csv
```

If the prediction file only contains IDs and model outputs, attach references
from the hosted dataset during scoring:

```bash
ANTHROPIC_API_KEY=... python evaluate.py \
  --predictions-file outputs/model_predictions.jsonl \
  --dataset luckyorangepepper/anon-pepper-72 \
  --split test \
  --output-file outputs/model_scores.jsonl \
  --summary-csv outputs/model_summary.csv
```

By default, the scorer uses `claude-opus-4-5` and automatically selects the saved
`text_qa`, `hypothesis_generation`, or `image_caption` rubric from
`evaluation/rubrics/`.

## Score Image Caption Outputs

Image-caption predictions can use this schema directly:

```json
{"paper_id":"...","image_id":"...","kind":"SEM-BSE","reference_caption":"...","predicted_caption":"...","image_path":"..."}
```

See `examples/image_caption_predictions.jsonl` for a minimal image-caption row.

Then run:

```bash
ANTHROPIC_API_KEY=... python evaluate.py \
  --answers-file outputs/image_predictions.jsonl \
  --output-file outputs/image_scores.jsonl \
  --summary-csv outputs/image_summary.csv \
  --rubric image_caption
```

`--rubric auto` also selects `image_caption` when it sees `reference_caption`,
`predicted_caption`, `image_path`, or modality labels such as `SEM`, `XRD`,
`EDS`, or `TGA`.

For image prediction files that only contain IDs plus `predicted_caption`, use
the same dataset-join pattern:

```bash
ANTHROPIC_API_KEY=... python evaluate.py \
  --predictions-file outputs/image_predictions.jsonl \
  --dataset luckyorangepepper/anon-pepper-72 \
  --split test \
  --rubric image_caption \
  --output-file outputs/image_scores.jsonl
```

Summarize an existing score file:

```bash
python evaluation/summarize_scores.py --scores-file outputs/model_scores.jsonl
```

The summarizer also accepts older image-judge files that store scores as
`llm_score`.
