# CKA Analysis

This folder contains the Qwen2-VL representation-similarity analysis used for
the paper. The CKA experiment is Qwen-only: it compares the base Qwen2-VL-7B
model with the MATRIX-QPT text-SFT, vision-SFT, and full-SFT adapters.

The released scripts do not commit generated hidden-state tensors, model
checkpoints, or CKA result files. Those are regenerated from the hosted MATRIX
data and the finetuned LoRA adapters.

## Inputs

- MATRIX data: `luckyorangepepper/anon-pepper-72`
- Qwen base model: `Qwen/Qwen2-VL-7B`
- Qwen processor for image-conditioned prompts: `Qwen/Qwen2-VL-7B-Instruct`
- Adapter paths: edit `configs/models_qwen.yaml` if your run names differ from
  the defaults.

## Reproduce

From the repository root:

```bash
bash scripts/prepare_matrix_data.sh
bash scripts/reproduce_cka.sh
```

`scripts/reproduce_cka.sh` extracts hidden states for three Qwen prompt sets:
text prompts, image-task text without image tensors, and image-conditioned
inputs with the real image plus prompt. It then computes layerwise linear CKA
and writes `cka/results/paper/cka_image_conditioned_summary.csv`.

The paper CKA claim should be read as Qwen-only. No Llama CKA experiment is
included in this artifact.
