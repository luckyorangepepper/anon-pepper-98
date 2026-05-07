from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def add_if_present(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def build_generate_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "inference/generate_predictions.py",
        "--model-path",
        args.model_path,
        "--output-file",
        str(args.predictions_file),
        "--task",
        args.task,
        "--split",
        args.split,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--num-beams",
        str(args.num_beams),
        "--torch-dtype",
        args.torch_dtype,
        "--device-map",
        args.device_map,
        "--caption-prompt",
        args.caption_prompt,
        "--flush-every",
        str(args.flush_every),
    ]
    add_if_present(cmd, "--base-model", args.base_model)
    add_if_present(cmd, "--processor-path", args.processor_path)
    add_if_present(cmd, "--input-file", args.input_file)
    add_if_present(cmd, "--dataset", args.dataset)
    add_if_present(cmd, "--hf-config", args.hf_config)
    add_if_present(cmd, "--image-root", args.image_root)
    add_if_present(cmd, "--max-samples", args.max_samples)
    if args.no_context:
        cmd.append("--no-context")
    if args.skip_missing_images:
        cmd.append("--skip-missing-images")
    if not args.trust_remote_code:
        cmd.append("--no-trust-remote-code")
    return cmd


def build_score_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "evaluate.py",
        "--answers-file",
        str(args.predictions_file),
        "--output-file",
        str(args.scores_file),
        "--rubric",
        args.rubric,
        "--judge-model",
        args.judge_model,
        "--batch-size",
        str(args.batch_size),
        "--max-tokens",
        str(args.max_tokens),
        "--max-retries",
        str(args.max_retries),
    ]
    add_if_present(cmd, "--summary-csv", args.summary_csv)
    add_if_present(cmd, "--api-key", args.api_key)
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MATRIX predictions and score them with the Opus judge.")

    parser.add_argument("--model-path", required=True, help="HF model ID, local model path, or LoRA adapter path.")
    parser.add_argument("--base-model", default=None, help="Base model for LoRA adapters; auto-read from adapter_config when possible.")
    parser.add_argument("--processor-path", default=None, help="Processor/tokenizer repo or path.")
    parser.add_argument("--input-file", type=Path, default=None, help="Local input JSON/JSONL. If omitted, --dataset is used.")
    parser.add_argument("--dataset", default="luckyorangepepper/anon-pepper-72")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-root", type=Path, default=None, help="Root for resolving relative image paths.")
    parser.add_argument("--task", choices=["auto", "text", "caption"], default="auto")
    parser.add_argument("--caption-prompt", default="What is a caption that is suitable for this image?")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--skip-missing-images", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")

    parser.add_argument("--predictions-file", type=Path, default=Path("outputs/model_predictions.jsonl"))
    parser.add_argument("--scores-file", type=Path, default=Path("outputs/model_scores.jsonl"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/model_summary.csv"))
    parser.add_argument("--rubric", default="auto", help="auto, text_qa, hypothesis_generation, image_caption, or a rubric path.")
    parser.add_argument("--judge-model", default="claude-opus-4-5")
    parser.add_argument("--api-key", default=None, help="Optional Anthropic API key. Prefer ANTHROPIC_API_KEY in the environment.")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--skip-score", action="store_true", help="Only generate predictions.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.predictions_file.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_score:
        args.scores_file.parent.mkdir(parents=True, exist_ok=True)
        if args.summary_csv:
            args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    generate_cmd = build_generate_cmd(args)
    print("Running inference:")
    print(" ".join(generate_cmd))
    subprocess.run(generate_cmd, check=True)

    if args.skip_score:
        return

    score_cmd = build_score_cmd(args)
    print("Running judge scoring:")
    print(" ".join(score_cmd))
    subprocess.run(score_cmd, check=True)


if __name__ == "__main__":
    main()
