# MATRIX Evaluation Artifact

Minimal evaluation code for the anonymized MATRIX dataset artifact.

Dataset:

```text
https://huggingface.co/datasets/luckyorangepepper/anon-pepper-72
```

Judge:

```text
claude-opus-4-5
```

- `evaluation/`: Opus judge scoring and fixed rubrics.
- `finetune/`: helper for exporting finetuning splits from Hugging Face.
- `examples/`: minimal accepted prediction schemas.

## Install

```bash
pip install -r requirements.txt
```

## Export Finetune Data

```bash
python finetune/export_dataset.py \
  --split train \
  --output finetune/exports/train.jsonl \
  --image-dir finetune/exports/images/train
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
