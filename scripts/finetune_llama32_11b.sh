#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-full}"  # text, vision, or full
RUN_NAME="${RUN_NAME:-llama32-${MODE}-sft}"
TEXT_ROOT="${TEXT_ROOT:-finetune/exports}"
TRAIN_PATH="${TRAIN_PATH:-finetune/exports/train.jsonl}"
VAL_PATH="${VAL_PATH:-finetune/exports/val.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-finetune/exports/images}"

case "${MODE}" in
  text)
    MIX_ARGS=("${TEXT_ROOT}")
    ;;
  vision)
    MIX_ARGS=("captions")
    ;;
  full)
    MIX_ARGS=("${TEXT_ROOT}" "captions")
    ;;
  *)
    echo "MODE must be text, vision, or full" >&2
    exit 2
    ;;
esac

python finetune/finetune_llama32_vision.py \
  --run-name "${RUN_NAME}" \
  --model-name meta-llama/Llama-3.2-11B-Vision \
  --qlora \
  --train-path "${TRAIN_PATH}" \
  --val-path "${VAL_PATH}" \
  --images-root "${IMAGE_ROOT}" \
  --text-mix-root "${MIX_ARGS[@]}" \
  --num-epochs "${NUM_EPOCHS:-2}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --grad-accum "${GRAD_ACCUM:-8}" \
  --lr "${LR:-2e-5}" \
  --lora-rank "${LORA_RANK:-8}" \
  --lora-alpha "${LORA_ALPHA:-32}" \
  --report-to "${REPORT_TO:-none}"
