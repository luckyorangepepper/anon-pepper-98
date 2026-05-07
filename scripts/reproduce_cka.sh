#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="${INPUT_FILE:-finetune/exports/test.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-finetune/exports/images/test}"
MAX_TEXT_EXAMPLES="${MAX_TEXT_EXAMPLES:-220}"
MAX_IMAGE_EXAMPLES="${MAX_IMAGE_EXAMPLES:-250}"

python cka/extract_hidden_states.py \
  --family qwen \
  --model-config cka/configs/models_qwen.yaml \
  --layers-config cka/configs/layers_qwen.yaml \
  --input-file "${INPUT_FILE}" \
  --source-filter text \
  --output-dir cka/results/hidden_states_text_prompts \
  --max-examples "${MAX_TEXT_EXAMPLES}"

python cka/compute_cka.py \
  --layers-config cka/configs/layers_qwen.yaml \
  --hidden-states-dir cka/results/hidden_states_text_prompts \
  --source text \
  --output-file cka/results/cka_prompt_only/qwen_text_cka_results.json

python cka/extract_hidden_states.py \
  --family qwen \
  --model-config cka/configs/models_qwen.yaml \
  --layers-config cka/configs/layers_qwen.yaml \
  --input-file "${INPUT_FILE}" \
  --source-filter image \
  --output-dir cka/results/hidden_states_image_prompt_only \
  --max-examples "${MAX_IMAGE_EXAMPLES}"

python cka/compute_cka.py \
  --layers-config cka/configs/layers_qwen.yaml \
  --hidden-states-dir cka/results/hidden_states_image_prompt_only \
  --source image \
  --output-file cka/results/cka_prompt_only/qwen_image_prompt_only_cka_results.json

python cka/extract_image_conditioned_hidden_states.py \
  --model-config cka/configs/models_qwen.yaml \
  --layers-config cka/configs/layers_qwen.yaml \
  --input-file "${INPUT_FILE}" \
  --image-root "${IMAGE_ROOT}" \
  --output-dir cka/results/hidden_states_image_conditioned \
  --max-examples "${MAX_IMAGE_EXAMPLES}"

python cka/compute_cka.py \
  --layers-config cka/configs/layers_qwen.yaml \
  --hidden-states-dir cka/results/hidden_states_image_conditioned \
  --source image_conditioned \
  --output-file cka/results/cka_image_conditioned/qwen_image_conditioned_cka_results.json

python cka/generate_paper_artifacts.py \
  --results \
    cka/results/cka_prompt_only/qwen_text_cka_results.json \
    cka/results/cka_prompt_only/qwen_image_prompt_only_cka_results.json \
    cka/results/cka_image_conditioned/qwen_image_conditioned_cka_results.json \
  --family Qwen2-VL-7B \
  --output-dir cka/results/paper \
  --output-name cka_image_conditioned_summary.csv
