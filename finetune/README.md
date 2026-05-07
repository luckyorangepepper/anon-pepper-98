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

## Paper Post-Training Runs

After exporting the data, launch Qwen runs with:

```bash
MODE=text RUN_NAME=qwen-text-sft bash scripts/finetune_qwen.sh
MODE=vision RUN_NAME=qwen-vision-sft bash scripts/finetune_qwen.sh
MODE=full RUN_NAME=qwen-full-sft bash scripts/finetune_qwen.sh
```

Launch Llama-3.2-11B-Vision runs with:

```bash
MODE=text RUN_NAME=llama32-text-sft bash scripts/finetune_llama32_11b.sh
MODE=vision RUN_NAME=llama32-vision-sft bash scripts/finetune_llama32_11b.sh
MODE=full RUN_NAME=llama32-full-sft bash scripts/finetune_llama32_11b.sh
```

The scripts write LoRA checkpoints under `finetune/runs/<run-name>/`. The CKA
config expects the Qwen adapters at `qwen-text-sft`, `qwen-vision-sft`, and
`qwen-full-sft`; edit `cka/configs/models_qwen.yaml` if you use different names.
