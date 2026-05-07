#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-luckyorangepepper/anon-pepper-72}"
EXPORT_ROOT="${EXPORT_ROOT:-finetune/exports}"

python finetune/export_dataset.py \
  --dataset "${DATASET}" \
  --split train \
  --output "${EXPORT_ROOT}/train.jsonl" \
  --image-dir "${EXPORT_ROOT}/images/train"

python finetune/export_dataset.py \
  --dataset "${DATASET}" \
  --split validation \
  --output "${EXPORT_ROOT}/val.jsonl" \
  --image-dir "${EXPORT_ROOT}/images/val"

python finetune/export_dataset.py \
  --dataset "${DATASET}" \
  --split test \
  --output "${EXPORT_ROOT}/test.jsonl" \
  --image-dir "${EXPORT_ROOT}/images/test"
