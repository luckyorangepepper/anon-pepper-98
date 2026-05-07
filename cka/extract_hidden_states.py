from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoTokenizer


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


def clean_question(text: str) -> str:
    return (
        text.replace("{image}", "")
        .replace("<image_placeholder>", "")
        .replace("<image>", "")
        .strip()
    )


def examples_from_rows(
    rows: list[dict[str, Any]],
    max_examples: int | None,
    source_filter: str,
) -> list[dict[str, str]]:
    examples = []
    for index, row in enumerate(rows):
        image_row = is_image_row(row)
        if source_filter == "text" and image_row:
            continue
        if source_filter == "image" and not image_row:
            continue
        text = first_present(row, ["question", "prompt", "input_prompt"])
        if not text:
            continue
        task = first_present(row, ["type", "task_type", "kind", "question_type", "modality"]) or "unknown"
        source = first_present(row, ["dataset_id", "paper_id", "qid", "id"]) or str(index)
        examples.append({"text": clean_question(text), "task": task, "source": source})
        if max_examples is not None and len(examples) >= max_examples:
            break
    if not examples:
        raise ValueError("No rows with a question/prompt were found.")
    return examples


def load_model_class(family: str):
    if family == "qwen":
        from transformers import Qwen2VLForConditionalGeneration

        return Qwen2VLForConditionalGeneration
    raise ValueError(f"Unsupported family: {family}")


def load_model_and_tokenizer(family: str, model_cfg: dict[str, Any], torch_dtype: torch.dtype):
    model_class = load_model_class(family)
    base_path = model_cfg["path"]
    adapter_path = model_cfg.get("adapter")

    model = model_class.from_pretrained(
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

    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def extract_hidden_states(model, tokenizer, examples: list[dict[str, str]], layers: list[int], max_length: int):
    hidden_states = {layer: [] for layer in layers}
    device = next(model.parameters()).device

    for index, example in enumerate(examples):
        if index % 50 == 0:
            print(f"Processing example {index}/{len(examples)}")
        inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        sequence_length = outputs.hidden_states[0].shape[1]
        pool_tokens = min(25, sequence_length)
        for layer in layers:
            pooled = outputs.hidden_states[layer + 1][0, -pool_tokens:, :].mean(dim=0).float().cpu()
            hidden_states[layer].append(pooled)

    return {layer: torch.stack(values) for layer, values in hidden_states.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pooled hidden states for MATRIX CKA analysis.")
    parser.add_argument("--family", choices=["qwen"], default="qwen")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--layers-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--dataset", default="luckyorangepepper/anon-pepper-72")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-examples", type=int, default=320)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--source-filter", choices=["all", "text", "image"], default="all")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.torch_dtype]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    layers = yaml.safe_load(args.layers_config.read_text(encoding="utf-8"))["layers"]
    examples = examples_from_rows(load_rows(args), args.max_examples, args.source_filter)
    torch.save(
        {
            "tasks": [example["task"] for example in examples],
            "sources": [example["source"] for example in examples],
            "texts": [example["text"] for example in examples],
        },
        args.output_dir / "labels.pt",
    )

    for model_name, cfg in model_cfg.items():
        missing_layers = [layer for layer in layers if not (args.output_dir / f"{model_name}_layer{layer}.pt").exists()]
        if not missing_layers:
            print(f"Skipping {model_name}; all requested layers already exist.")
            continue
        print(f"Loading {model_name}")
        model, tokenizer = load_model_and_tokenizer(args.family, cfg, dtype)
        tensors = extract_hidden_states(model, tokenizer, examples, missing_layers, args.max_length)
        for layer, tensor in tensors.items():
            out_path = args.output_dir / f"{model_name}_layer{layer}.pt"
            torch.save(tensor, out_path)
            print(f"Saved {out_path} {tuple(tensor.shape)}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
