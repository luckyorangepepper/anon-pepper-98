#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-full}"  # text, vision, or full
RUN_NAME="${RUN_NAME:-qwen-${MODE}-sft}"
TEXT_ROOT="${TEXT_ROOT:-finetune/exports}"
TRAIN_PATH="${TRAIN_PATH:-finetune/exports/train.jsonl}"
VAL_PATH="${VAL_PATH:-finetune/exports/val.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-finetune/exports/images}"

case "${MODE}" in
  text)
    MIX_ARGS=("${TEXT_ROOT}")
    LR_DEFAULT="5e-6"
    GRAD_ACCUM_DEFAULT="32"
    LR_SCHEDULER_DEFAULT="linear"
    NUM_WARMUP_STEPS_DEFAULT="0"
    WARMUP_RATIO_DEFAULT="0.03"
    ;;
  vision)
    MIX_ARGS=("captions")
    LR_DEFAULT="2e-5"
    GRAD_ACCUM_DEFAULT="1"
    LR_SCHEDULER_DEFAULT="cosine"
    NUM_WARMUP_STEPS_DEFAULT="100"
    WARMUP_RATIO_DEFAULT="0.0"
    ;;
  full)
    MIX_ARGS=("${TEXT_ROOT}" "captions")
    LR_DEFAULT="2e-5"
    GRAD_ACCUM_DEFAULT="1"
    LR_SCHEDULER_DEFAULT="cosine"
    NUM_WARMUP_STEPS_DEFAULT="100"
    WARMUP_RATIO_DEFAULT="0.0"
    ;;
  *)
    echo "MODE must be text, vision, or full" >&2
    exit 2
    ;;
esac

python finetune/finetune_datamix.py \
  --run-name "${RUN_NAME}" \
  --model-name Qwen/Qwen2-VL-7B \
  --processor-preference instruct \
  --fp8 \
  --train-path "${TRAIN_PATH}" \
  --val-path "${VAL_PATH}" \
  --images-root "${IMAGE_ROOT}" \
  --text-mix-root "${MIX_ARGS[@]}" \
  --num-epochs "${NUM_EPOCHS:-2}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --grad-accum "${GRAD_ACCUM:-${GRAD_ACCUM_DEFAULT}}" \
  --lr "${LR:-${LR_DEFAULT}}" \
  --lr-scheduler "${LR_SCHEDULER:-${LR_SCHEDULER_DEFAULT}}" \
  --num-warmup-steps "${NUM_WARMUP_STEPS:-${NUM_WARMUP_STEPS_DEFAULT}}" \
  --warmup-ratio "${WARMUP_RATIO:-${WARMUP_RATIO_DEFAULT}}" \
  --lora-rank "${LORA_RANK:-8}" \
  --lora-alpha "${LORA_ALPHA:-32}" \
  --report-to "${REPORT_TO:-none}"
