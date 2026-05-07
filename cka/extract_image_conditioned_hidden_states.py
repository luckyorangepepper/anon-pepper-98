from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return list(raw.values())
    raise ValueError(f"Unsupported JSON structure in {path}")


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_file:
        return read_json_or_jsonl(args.input_file)

    from datasets import load_dataset

    if args.hf_config:
        dataset = load_dataset(args.dataset, args.hf_config, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)
    return [dict(row) for row in dataset]


def first_present(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return str(value)
    return ""


def clean_question(text: str) -> str:
    return (
        text.replace("{image}", "")
        .replace("<image_placeholder>", "")
        .replace("<image>", "")
        .strip()
    )


def is_image_row(row: dict[str, Any]) -> bool:
    question = first_present(row, ["question", "prompt", "input_prompt"])
    task_type = first_present(row, ["type", "task_type"]).lower()
    return (
        task_type in {"vision", "image", "caption"}
        or row.get("image") is not None
        or bool(first_present(row, ["image_path", "image_filename", "path"]))
        or "{image}" in question
        or "<image>" in question
        or "<image_placeholder>" in question
    )


def resolve_image_path(row: dict[str, Any], image_root: Path | None) -> Path | None:
    image_ref = first_present(row, ["image_path", "image_filename", "path"])
    if not image_ref:
        return None

    image_path = Path(image_ref)
    candidates = [image_path]
    if image_root is not None:
        candidates.extend([image_root / image_path, image_root / image_path.name])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def examples_from_rows(
    rows: list[dict[str, Any]],
    image_root: Path | None,
    max_examples: int | None,
) -> list[dict[str, Any]]:
    examples = []
    for index, row in enumerate(rows):
        if not is_image_row(row):
            continue

        question = clean_question(first_present(row, ["question", "prompt", "input_prompt"]))
        if not question:
            continue

        image_path = resolve_image_path(row, image_root)
        image_obj = row.get("image")
        if image_path is None and image_obj is None:
            continue

        qid = first_present(row, ["qid", "id", "image_id"]) or str(index)
        task = first_present(row, ["kind", "question_type", "modality", "type", "task_type"]) or "image"
        examples.append({"qid": qid, "task": task, "text": question, "image_path": image_path, "image": image_obj})
        if max_examples is not None and len(examples) >= max_examples:
            break

    if not examples:
        raise ValueError("No image rows with question text and image data were found.")
    return examples


def load_model_and_processor(model_cfg: dict[str, Any], processor_name: str, torch_dtype: torch.dtype):
    base_path = model_cfg["path"]
    adapter_path = model_cfg.get("adapter")

    processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        adapter = Path(adapter_path)
        if not adapter.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter))
        model = model.merge_and_unload()
    model.eval()
    return model, processor


def load_image(example: dict[str, Any]) -> Image.Image:
    if example.get("image_path") is not None:
        with Image.open(example["image_path"]) as handle:
            return handle.convert("RGB")

    image = example["image"]
    if isinstance(image, dict) and image.get("path"):
        with Image.open(image["path"]) as handle:
            return handle.convert("RGB")
    if hasattr(image, "convert"):
        return image.convert("RGB")
    raise ValueError(f"Unsupported image object for {example['qid']}")


def build_inputs(processor, question: str, image: Image.Image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], padding=True, return_tensors="pt")


def move_inputs(inputs, device: torch.device, torch_dtype: torch.dtype):
    moved = {}
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            moved[key] = value
        elif torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=torch_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def extract_hidden_states(model, processor, examples: list[dict[str, Any]], layers: list[int], torch_dtype: torch.dtype):
    hidden_states = {layer: [] for layer in layers}
    device = next(model.parameters()).device

    for example in tqdm(examples, desc="image-conditioned hidden states"):
        image = load_image(example)
        inputs = move_inputs(build_inputs(processor, example["text"], image), device, torch_dtype)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        sequence_length = outputs.hidden_states[0].shape[1]
        pool_tokens = min(25, sequence_length)
        for layer in layers:
            pooled = outputs.hidden_states[layer + 1][0, -pool_tokens:, :].mean(dim=0).float().cpu()
            hidden_states[layer].append(pooled)

    return {layer: torch.stack(values) for layer, values in hidden_states.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Qwen image-conditioned pooled hidden states for CKA.")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--layers-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--dataset", default="luckyorangepepper/anon-pepper-72")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--processor-name", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--models", nargs="+", default=["base", "text-sft", "vision-sft", "full-sft"])
    parser.add_argument("--max-examples", type=int, default=250)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.torch_dtype]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    models_cfg = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    layers = yaml.safe_load(args.layers_config.read_text(encoding="utf-8"))["layers"]
    examples = examples_from_rows(load_rows(args), args.image_root, args.max_examples)
    torch.save(
        {
            "tasks": [example["task"] for example in examples],
            "qids": [example["qid"] for example in examples],
            "texts": [example["text"] for example in examples],
            "image_paths": [str(example["image_path"]) for example in examples],
        },
        args.output_dir / "labels.pt",
    )

    for model_name in args.models:
        if model_name not in models_cfg:
            raise KeyError(f"Unknown model {model_name}; available: {sorted(models_cfg)}")
        missing_layers = [
            layer
            for layer in layers
            if args.force or not (args.output_dir / f"{model_name}_layer{layer}.pt").exists()
        ]
        if not missing_layers:
            print(f"Skipping {model_name}; all requested layers already exist.")
            continue

        print(f"Loading {model_name}")
        model, processor = load_model_and_processor(models_cfg[model_name], args.processor_name, dtype)
        tensors = extract_hidden_states(model, processor, examples, missing_layers, dtype)
        for layer, tensor in tensors.items():
            out_path = args.output_dir / f"{model_name}_layer{layer}.pt"
            torch.save(tensor, out_path)
            print(f"Saved {out_path} {tuple(tensor.shape)}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
