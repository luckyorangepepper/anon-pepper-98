from __future__ import annotations

import argparse
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "luckyorangepepper/anon-pepper-72"
DEFAULT_CAPTION_PROMPT = "What is a caption that is suitable for this image?"
IMAGE_CAPTION_KINDS = {
    "eds",
    "sem",
    "sem-bse",
    "sem-se",
    "bse",
    "se",
    "tga",
    "xrd",
    "vision",
    "image",
    "experimental",
    "caption",
}
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class Runtime:
    torch: Any
    model: Any
    processor: Any
    tokenizer: Any
    decode_tokenizer: Any | None


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

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install `datasets` to load rows with --dataset.") from exc

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


def row_identifier(row: dict[str, Any], fallback_index: int) -> str:
    explicit = first_present(row, ["qid", "id"])
    if explicit:
        return explicit

    paper_id = first_present(row, ["paper_id"])
    image_id = first_present(row, ["image_id"])
    if paper_id and image_id:
        return f"{paper_id}:{image_id}"
    if image_id:
        return image_id
    if paper_id:
        return paper_id
    return str(fallback_index)


def choose_task(row: dict[str, Any], task: str) -> str:
    if task in {"text", "caption"}:
        return task

    task_type = str(row.get("type") or row.get("task_type") or "").lower()
    kind = str(row.get("kind") or row.get("question_type") or row.get("modality") or "").lower()
    if row.get("reference_caption") or row.get("predicted_caption") or row.get("caption"):
        return "caption"
    if row.get("image") is not None or first_present(row, ["image_path", "image_filename", "path"]):
        return "caption"
    if task_type == "vision" or any(label in f"{task_type} {kind}" for label in IMAGE_CAPTION_KINDS):
        return "caption"
    return "text"


def answer_for_task(row: dict[str, Any], task: str) -> str:
    if task == "caption":
        return first_present(
            row,
            ["reference_caption", "caption", "description", "answer", "ground_truth_answer", "reference_answer"],
        )
    return first_present(row, ["answer", "ground_truth_answer", "reference_answer", "reference_caption"])


def prompt_for_task(row: dict[str, Any], task: str, include_context: bool, caption_prompt: str) -> str:
    prompt = first_present(row, ["question", "prompt", "input_prompt"])
    if task == "caption" and not prompt:
        prompt = caption_prompt

    context = first_present(row, ["context", "paper_context"])
    if include_context and context:
        return f"{context}\n\n{prompt}".strip()
    return prompt


def candidate_paths(image_ref: str, row: dict[str, Any], image_root: Path | None) -> list[Path]:
    if not image_ref:
        return []

    image_path = Path(image_ref)
    has_suffix = image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    paper_id = first_present(row, ["paper_id"])
    candidates: list[Path] = []

    def add(path: Path) -> None:
        candidates.append(path)
        if not has_suffix:
            for suffix in [".png", ".jpg", ".jpeg", ".webp"]:
                candidates.append(path.with_suffix(suffix))

    add(image_path)
    if image_root is not None:
        add(image_root / image_path)
        if paper_id:
            add(image_root / paper_id / image_path)

    return list(dict.fromkeys(candidates))


def resolve_image(row: dict[str, Any], image_root: Path | None) -> tuple[Any | None, str]:
    image_obj = row.get("image")
    image_path_hint = first_present(row, ["image_path", "image_filename", "path", "image_id"])

    if image_obj is not None:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install `pillow` to run image inference.") from exc

        if hasattr(image_obj, "convert"):
            return image_obj.convert("RGB"), image_path_hint
        if isinstance(image_obj, dict):
            if image_obj.get("path"):
                image_path_hint = str(image_obj["path"])
            if image_obj.get("bytes"):
                return Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB"), image_path_hint

    for path in candidate_paths(image_path_hint, row, image_root):
        if path.exists():
            from PIL import Image

            return Image.open(path).convert("RGB"), str(path)

    return None, image_path_hint


def import_transformers() -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    imported: dict[str, Any] = {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoProcessor": AutoProcessor,
        "AutoTokenizer": AutoTokenizer,
    }
    for name in ["AutoModelForImageTextToText", "AutoModelForVision2Seq", "Qwen2VLForConditionalGeneration"]:
        try:
            module = __import__("transformers", fromlist=[name])
            imported[name] = getattr(module, name)
        except (ImportError, AttributeError):
            imported[name] = None
    try:
        from transformers import MistralCommonBackend

        imported["MistralCommonBackend"] = MistralCommonBackend
    except ImportError:
        imported["MistralCommonBackend"] = None
    return imported


def torch_dtype_from_arg(torch: Any, value: str) -> Any:
    if value == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def maybe_qwen_instruct_processor(source: str) -> str:
    lowered = source.lower()
    if "qwen2-vl" in lowered and "instruct" not in lowered:
        return source.rstrip("/") + "-Instruct"
    return source


def adapter_base_model(model_path: Path, fallback: str | None) -> str | None:
    config_path = model_path / "adapter_config.json"
    if not config_path.exists():
        return fallback
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback
    return fallback or config.get("base_model_name_or_path")


def load_first_model(loaders: list[Any], source: str, kwargs: dict[str, Any]) -> Any:
    errors = []
    for loader in loaders:
        if loader is None:
            continue
        try:
            return loader.from_pretrained(source, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on model family
            errors.append(f"{loader.__name__}: {exc}")
    raise RuntimeError(f"Could not load model from {source}:\n" + "\n".join(errors))


def tokenizer_from_processor(processor: Any, auto_tokenizer: Any, source: str, trust_remote_code: bool) -> Any:
    if hasattr(processor, "tokenizer"):
        return processor.tokenizer
    if hasattr(processor, "encode"):
        return processor
    return auto_tokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)


def align_tokenizer_and_model(tokenizer: Any, model: Any) -> None:
    special_tokens = {}
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None)
    if getattr(tokenizer, "eos_token", None) is None:
        special_tokens["eos_token"] = DEFAULT_EOS_TOKEN
    if getattr(tokenizer, "bos_token", None) is None:
        special_tokens["bos_token"] = DEFAULT_BOS_TOKEN
    if getattr(tokenizer, "unk_token", None) is None:
        special_tokens["unk_token"] = DEFAULT_UNK_TOKEN

    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        return
    if num_new_tokens > 0 or input_embeddings.weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))


def load_runtime(args: argparse.Namespace) -> Runtime:
    imported = import_transformers()
    torch = imported["torch"]
    dtype = torch_dtype_from_arg(torch, args.torch_dtype)

    model_path = Path(args.model_path)
    is_local = model_path.exists()
    is_adapter = is_local and (model_path / "adapter_config.json").exists()
    model_source = str(model_path) if is_local else args.model_path
    base_source = adapter_base_model(model_path, args.base_model) if is_adapter else args.base_model
    weights_source = base_source if is_adapter and base_source else model_source
    processor_source = args.processor_path or maybe_qwen_instruct_processor(base_source or model_source)

    common_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
    }
    if args.device_map != "none":
        common_kwargs["device_map"] = args.device_map

    print(f"Loading processor: {processor_source}")
    try:
        processor = imported["AutoProcessor"].from_pretrained(
            processor_source,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception:
        processor = imported["AutoTokenizer"].from_pretrained(
            processor_source,
            trust_remote_code=args.trust_remote_code,
        )

    tokenizer = tokenizer_from_processor(
        processor,
        imported["AutoTokenizer"],
        processor_source,
        args.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    loaders = [
        imported["AutoModelForImageTextToText"],
        imported["AutoModelForVision2Seq"],
        imported["Qwen2VLForConditionalGeneration"],
        imported["AutoModelForCausalLM"],
    ]
    print(f"Loading weights: {weights_source}")
    model = load_first_model(loaders, weights_source, common_kwargs)
    align_tokenizer_and_model(tokenizer, model)

    if is_adapter:
        from peft import PeftModel

        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, str(model_path))

    if args.device_map == "none":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True
    model.eval()

    decode_tokenizer = None
    if "ministral" in weights_source.lower() and imported["MistralCommonBackend"] is not None:
        decode_tokenizer = imported["MistralCommonBackend"].from_pretrained(weights_source)

    return Runtime(torch=torch, model=model, processor=processor, tokenizer=tokenizer, decode_tokenizer=decode_tokenizer)


def message_candidates(prompt: str, has_image: bool, image_path: str | None) -> list[list[dict[str, Any]]]:
    if has_image:
        image_with_path = {"type": "image", "image": image_path} if image_path else {"type": "image"}
        return [
            [{"role": "user", "content": [image_with_path, {"type": "text", "text": prompt}]}],
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            [{"role": "user", "content": f"<image>\n{prompt}"}],
        ]
    return [
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        [{"role": "user", "content": prompt}],
    ]


def render_prompt(processor: Any, prompt: str, has_image: bool, image_path: str | None) -> str:
    if hasattr(processor, "apply_chat_template"):
        for messages in message_candidates(prompt, has_image, image_path):
            try:
                rendered = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                try:
                    rendered = processor.apply_chat_template(messages, add_generation_prompt=True)
                except Exception:
                    continue
            except Exception:
                continue
            if isinstance(rendered, str) and rendered.strip():
                return rendered

    if has_image:
        return f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def to_device(batch: dict[str, Any], torch: Any, device: Any, dtype: Any) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            moved[key] = value
            continue
        if value.dtype.is_floating_point:
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def decode_generated(runtime: Runtime, output_ids: Any, input_len: int) -> str:
    generated_ids = output_ids[:, input_len:] if output_ids.shape[-1] > input_len else output_ids
    if runtime.decode_tokenizer is not None:
        return runtime.decode_tokenizer.decode(generated_ids[0].tolist()).strip()
    decoder = runtime.processor if hasattr(runtime.processor, "batch_decode") else runtime.tokenizer
    return decoder.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()


def generate_one(
    runtime: Runtime,
    *,
    prompt: str,
    image: Any | None,
    image_path: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_beams: int,
) -> str:
    rendered = render_prompt(runtime.processor, prompt, image is not None, image_path)
    if image is not None:
        inputs = runtime.processor(text=[rendered], images=[image], return_tensors="pt")
    else:
        tokenizer = runtime.tokenizer
        inputs = tokenizer([rendered], return_tensors="pt")

    inputs.pop("token_type_ids", None)
    device = next(runtime.model.parameters()).device
    dtype = next(runtime.model.parameters()).dtype
    inputs = to_device(inputs, runtime.torch, device, dtype)

    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_beams": num_beams,
    }
    if runtime.tokenizer.pad_token_id is not None:
        gen_kwargs["pad_token_id"] = runtime.tokenizer.pad_token_id
    if runtime.tokenizer.eos_token_id is not None:
        gen_kwargs["eos_token_id"] = runtime.tokenizer.eos_token_id
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with runtime.torch.no_grad():
        output_ids = runtime.model.generate(**inputs, **gen_kwargs)

    return decode_generated(runtime, output_ids, inputs["input_ids"].shape[-1]).strip()


def output_record(
    row: dict[str, Any],
    *,
    index: int,
    task: str,
    prompt: str,
    answer: str,
    prediction: str,
    image_path: str,
) -> dict[str, Any]:
    task_type = first_present(row, ["type", "task_type"]) or ("vision" if task == "caption" else "text")
    kind = first_present(row, ["kind", "question_type", "modality"])

    record: dict[str, Any] = {
        "qid": row_identifier(row, index),
        "type": task_type,
        "kind": kind,
        "question": prompt,
        "answer": answer,
    }
    for key in ["paper_id", "image_id", "dataset_id"]:
        if row.get(key) is not None:
            record[key] = str(row[key])

    if task == "caption":
        record["reference_caption"] = answer
        record["predicted_caption"] = prediction
        if image_path:
            record["image_path"] = image_path
    else:
        record["model_answer"] = prediction
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MATRIX prediction JSONL for later Opus scoring.")
    parser.add_argument("--model-path", required=True, help="HF model ID, full model path, or LoRA adapter path.")
    parser.add_argument("--base-model", default=None, help="Base model for LoRA adapters; auto-read from adapter_config when possible.")
    parser.add_argument("--processor-path", default=None, help="Processor/tokenizer repo or path. Defaults to the model/base source.")
    parser.add_argument("--input-file", type=Path, default=None, help="Local JSON/JSONL rows. If omitted, --dataset is loaded.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset to use when --input-file is omitted.")
    parser.add_argument("--hf-config", default=None, help="Optional Hugging Face dataset config.")
    parser.add_argument("--split", default="test", help="Dataset split to generate.")
    parser.add_argument("--image-root", type=Path, default=None, help="Root for resolving relative image paths.")
    parser.add_argument("--output-file", type=Path, required=True, help="JSONL prediction file to write.")
    parser.add_argument("--task", choices=["auto", "text", "caption"], default="auto")
    parser.add_argument("--caption-prompt", default=DEFAULT_CAPTION_PROMPT)
    parser.add_argument("--no-context", action="store_true", help="Do not prepend row context to prompts.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value, or 'none' to move model manually.")
    parser.add_argument("--skip-missing-images", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    rows = load_rows(args)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No input rows found.")

    runtime = load_runtime(args)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = lambda iterable, **_: iterable  # noqa: E731

    written = 0
    with args.output_file.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(tqdm(rows, desc="Generating predictions")):
            task = choose_task(row, args.task)
            prompt = prompt_for_task(row, task, not args.no_context, args.caption_prompt)
            if not prompt:
                print(f"Warning: skipping row {index}; no prompt/question found.")
                continue

            image = None
            image_path = ""
            if task == "caption":
                image, image_path = resolve_image(row, args.image_root)
                if image is None and not args.skip_missing_images:
                    raise FileNotFoundError(f"Could not resolve image for row {index}: {image_path}")
                if image is None:
                    print(f"Warning: skipping row {index}; image not found: {image_path}")
                    continue

            prediction = generate_one(
                runtime,
                prompt=prompt,
                image=image,
                image_path=image_path,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
            )
            record = output_record(
                row,
                index=index,
                task=task,
                prompt=prompt,
                answer=answer_for_task(row, task),
                prediction=prediction,
                image_path=image_path,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if args.flush_every > 0 and written % args.flush_every == 0:
                handle.flush()

    print(f"Wrote {written} predictions to {args.output_file}")


if __name__ == "__main__":
    main()
