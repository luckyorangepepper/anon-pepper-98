# Finetune Dataset

The finetuning data is hosted with the anonymized Hugging Face dataset:

```text
luckyorangepepper/anon-pepper-72
```

This folder is intentionally lightweight. It provides a helper to export any HF
split into local JSONL for Qwen/Llama post-training code, with optional image
materialization for multimodal SFT.

## Export A Split

```bash
python finetune/export_dataset.py \
  --split train \
  --output finetune/exports/train.jsonl \
  --image-dir finetune/exports/images/train
```

The JSONL rows use a simple schema:

```json
{"qid":"...","type":"vision","kind":"SEM-BSE","question":"...","answer":"...","image_path":"images/train/example.png"}
```

For text-only SFT, omit `--image-dir`; image objects are skipped unless an
existing `image_path` field is present.

## Common Splits

The export script accepts whatever splits are present on Hugging Face, typically:

- `train`
- `validation` or `val`
- `test`

Use `--dataset` to override the default dataset ID.
