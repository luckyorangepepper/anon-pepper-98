# Evaluation

This folder contains the fixed rubrics used for MATRIX evaluation.
See `PROVENANCE.md` for how these rubrics map to the experiment scorers used for
the paper tables.

Default judge:

```text
claude-opus-4-5
```

Default dataset:

```text
luckyorangepepper/anon-pepper-72
```

## Rubrics

- `rubrics/text_qa.md`: theory and research Q&A semantic scoring.
- `rubrics/hypothesis_generation.md`: hypothesis generation.
- `rubrics/image_caption.md`: image-caption semantic scoring.

## Scoring Command

```bash
ANTHROPIC_API_KEY=... python evaluate.py \
  --answers-file outputs/model_predictions.jsonl \
  --output-file outputs/model_scores.jsonl \
  --summary-csv outputs/model_summary.csv
```

The input can be JSONL or JSON. Text rows should include `question`, `answer`,
and `model_answer`. Image-caption rows may instead use `reference_caption` and
`predicted_caption`. Optional `type`, `kind`, `image_path`, and modality fields
are used by `--rubric auto` to choose the task-specific rubric.

Examples are provided in `../examples/text_predictions.jsonl` and
`../examples/image_caption_predictions.jsonl`.

If prediction rows only contain IDs and model outputs, pass `--dataset` and
`--split`; the evaluator joins references from Hugging Face before judging.

Use a fixed rubric when needed:

```bash
python evaluation/score_with_opus.py \
  --answers-file outputs/model_predictions.jsonl \
  --output-file outputs/model_scores.jsonl \
  --rubric hypothesis_generation
```
