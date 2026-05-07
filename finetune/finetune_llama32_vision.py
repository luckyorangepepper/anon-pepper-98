from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import ConcatDataset, Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

IGNORE_INDEX = -100
MAX_LENGTH = 3072
DEFAULT_EXPORT_ROOT = Path("finetune/exports")
DEFAULT_IMAGE_ROOT = DEFAULT_EXPORT_ROOT / "images"
DEFAULT_OUTPUT_DIR = Path("finetune/runs")


def first_present(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return str(value)
    return ""


def is_image_record(row: dict[str, Any]) -> bool:
    question = first_present(row, ["question", "prompt", "instruction"])
    task_type = first_present(row, ["type", "task_type"]).lower()
    return (
        task_type in {"vision", "image", "caption"}
        or bool(first_present(row, ["image_path", "image_filename", "path", "image_id"]))
        or row.get("image") is not None
        or "{image}" in question
        or "<image>" in question
        or "<image_placeholder>" in question
    )


class LatestCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        output_dir = Path(args.output_dir)
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        if not checkpoints:
            return
        latest = output_dir / "latest"
        try:
            latest.unlink(missing_ok=True) if latest.is_symlink() else shutil.rmtree(latest, ignore_errors=True)
            latest.symlink_to(checkpoints[-1].name)
        except FileExistsError:
            pass


def ensure_llama_image_size(image: Image.Image) -> Image.Image:
    max_dim = 560
    width, height = image.size
    if width > max_dim or height > max_dim:
        scale = max_dim / max(width, height)
        image = image.resize((int(width * scale), int(height * scale)), resample=Image.Resampling.BICUBIC)
        width, height = image.size
    if min(width, height) < 28:
        scale = 28 / float(min(width, height))
        image = image.resize((int(math.ceil(width * scale)), int(math.ceil(height * scale))), resample=Image.Resampling.BICUBIC)
    return image


class ImageCaptionJsonlDataset(Dataset):
    def __init__(self, jsonl_path: Path, processor: AutoProcessor, images_root: Path | None, prompt_text: str):
        self.processor = processor
        self.images_root = images_root
        self.prompt_text = prompt_text
        self.samples: list[dict[str, str]] = []

        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("deleted") or row.get("empty_caption") or row.get("empty_description"):
                    continue

                description = first_present(row, ["description", "answer", "caption", "reference_caption"])
                image_ref = first_present(row, ["image_path", "image_filename", "path", "image_id"])
                paper_id = first_present(row, ["paper_id"])
                if not description or not image_ref:
                    continue

                image_path = self.resolve_image_path(paper_id, image_ref)
                if image_path is None:
                    print(f"Warning: missing image for {jsonl_path.name} line {line_number}; skipping.")
                    continue
                self.samples.append({"image_path": str(image_path), "description": description})

        if not self.samples:
            raise ValueError(f"No usable image-caption samples found in {jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def resolve_image_path(self, paper_id: str, image_ref: str) -> Path | None:
        image_path = Path(image_ref)
        has_extension = image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        candidates: list[Path] = []

        def add(path: Path) -> None:
            candidates.append(path)
            if not has_extension:
                candidates.extend([path.with_suffix(".png"), path.with_suffix(".jpg"), path.with_suffix(".jpeg")])

        add(image_path)
        if self.images_root:
            add(self.images_root / image_path)
            if paper_id:
                add(self.images_root / paper_id / image_path)
        for candidate in dict.fromkeys(candidates):
            if candidate.exists():
                return candidate
        return None

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        with Image.open(sample["image_path"]) as handle:
            image = ensure_llama_image_size(handle.convert("RGB"))

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": sample["description"]}]},
        ]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full = self.processor(text=rendered, images=[image], add_special_tokens=False, return_tensors="pt")

        input_ids = full["input_ids"][0]
        labels = input_ids.clone()
        attention_mask = full["attention_mask"][0]

        user_messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt_text}]}]
        user_rendered = self.processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
        user = self.processor(text=user_rendered, images=[image], add_special_tokens=False, return_tensors="pt")
        user_len = min(user["input_ids"].shape[-1], labels.shape[-1])
        labels[:user_len] = IGNORE_INDEX

        eos_id = self.processor.tokenizer.eos_token_id
        if eos_id is not None:
            for pos in (input_ids == eos_id).nonzero(as_tuple=False).flatten().tolist():
                if pos >= user_len:
                    labels[pos] = eos_id

        if input_ids.shape[-1] > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]

        example: dict[str, torch.Tensor] = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        for key in ["pixel_values", "aspect_ratio_ids", "aspect_ratio_mask"]:
            if key in full:
                example[key] = full[key]
        if "cross_attention_mask" in full:
            cam = full["cross_attention_mask"].squeeze(0)
            example["cross_attention_mask"] = cam[:MAX_LENGTH] if cam.shape[0] > MAX_LENGTH else cam
        return example


class TextJsonlDataset(Dataset):
    def __init__(self, jsonl_path: Path, processor: AutoProcessor):
        self.processor = processor
        self.samples: list[dict[str, str]] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if is_image_record(row):
                    continue
                question = first_present(row, ["question", "prompt", "instruction"])
                answer = first_present(row, ["answer", "response", "output", "ground_truth_answer", "reference_answer"])
                if question and answer:
                    self.samples.append({"question": question, "answer": answer})
        if not self.samples:
            raise ValueError(f"No usable text samples found in {jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        messages = [
            {"role": "user", "content": [{"type": "text", "text": sample["question"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
        ]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        model_inputs = self.processor(text=rendered, return_tensors="pt")
        input_ids = model_inputs["input_ids"][0]
        labels = input_ids.clone()
        attention_mask = model_inputs["attention_mask"][0]

        user_messages = [{"role": "user", "content": [{"type": "text", "text": sample["question"]}]}]
        user_rendered = self.processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
        user_inputs = self.processor(text=user_rendered, return_tensors="pt")
        labels[: min(user_inputs["input_ids"].shape[-1], labels.shape[-1])] = IGNORE_INDEX

        if input_ids.shape[-1] > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


@dataclass
class LlamaVisionDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids = [item["input_ids"].clone() for item in instances]
        labels = [item["labels"].clone() for item in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        def pad8(tensor, value):
            remainder = tensor.size(1) % 8
            return tensor if remainder == 0 else torch.nn.functional.pad(tensor, (0, 8 - remainder), value=value)

        input_ids = pad8(input_ids, self.tokenizer.pad_token_id)
        labels = pad8(labels, IGNORE_INDEX)
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": input_ids.ne(self.tokenizer.pad_token_id)}

        if any("pixel_values" in item for item in instances):
            pixel_values = [item["pixel_values"] for item in instances if "pixel_values" in item]
            if pixel_values:
                fused = torch.cat(pixel_values, dim=0)
                batch["pixel_values"] = fused.to(torch.bfloat16) if fused.dtype == torch.float32 else fused
            for key in ["aspect_ratio_ids", "aspect_ratio_mask"]:
                values = [item[key] for item in instances if key in item]
                if values:
                    batch[key] = torch.cat(values, dim=0)

            cams = [item["cross_attention_mask"] for item in instances if "cross_attention_mask" in item]
            if cams:
                padded_seq_len = input_ids.shape[1]
                padded = []
                for cam in cams:
                    if cam.shape[0] < padded_seq_len:
                        pad_shape = (padded_seq_len - cam.shape[0],) + cam.shape[1:]
                        cam = torch.cat([cam, torch.zeros(pad_shape, dtype=cam.dtype)], dim=0)
                    padded.append(cam[:padded_seq_len])
                ref = padded[0]
                cam_index = 0
                all_cams = []
                for item in instances:
                    if "cross_attention_mask" in item:
                        all_cams.append(padded[cam_index])
                        cam_index += 1
                    else:
                        all_cams.append(torch.zeros((padded_seq_len,) + ref.shape[1:], dtype=ref.dtype))
                batch["cross_attention_mask"] = torch.stack(all_cams, dim=0)
        return batch


def find_jsonl_files(root: Path, split: str) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file() and root.suffix == ".jsonl":
        return [root]
    paths = []
    direct = root / f"{split}.jsonl"
    if direct.exists():
        paths.append(direct)
    paths.extend(sorted(root.rglob(f"{split}.jsonl")))
    return list(dict.fromkeys(paths))


def setup_datasets(args, processor):
    train_datasets: list[Dataset] = []
    val_datasets: list[Dataset] = []
    for entry in args.text_mix_root:
        if entry.lower() == "captions":
            train_datasets.append(ImageCaptionJsonlDataset(args.train_path, processor, args.images_root, args.prompt_text))
            val_datasets.append(ImageCaptionJsonlDataset(args.val_path, processor, args.images_root, args.prompt_text))
        else:
            root = Path(entry)
            train_datasets.extend(TextJsonlDataset(path, processor) for path in find_jsonl_files(root, "train"))
            val_datasets.extend(TextJsonlDataset(path, processor) for path in find_jsonl_files(root, "val"))
    if not train_datasets or not val_datasets:
        raise ValueError("No train/val datasets were created. Check --text-mix-root, --train-path, and --val-path.")
    train = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    val = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)
    print(f"[datamix] {len(train_datasets)} datasets, {len(train)} train, {len(val)} val samples")
    return train, val


def setup_model(args):
    quantization_config = None
    if args.qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    processor_repo = args.processor_name or (args.model_name if args.model_name.endswith("-Instruct") else args.model_name + "-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map=None,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if args.qlora:
        model = prepare_model_for_kbit_training(model)

    if args.freeze_vision:
        if hasattr(model, "model") and hasattr(model.model, "vision_model"):
            model.model.vision_model.requires_grad_(False)
        if hasattr(model, "model") and hasattr(model.model, "multi_modal_projector"):
            model.model.multi_modal_projector.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(processor_repo, trust_remote_code=True)
    tokenizer = processor.tokenizer
    tokenizer.model_max_length = MAX_LENGTH
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.peft_model:
        model = PeftModel.from_pretrained(model, args.peft_model, device_map="auto")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False
    return model, tokenizer, processor


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT for Llama-3.2-11B-Vision on MATRIX text and image-caption data.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-11B-Vision")
    parser.add_argument("--processor-name", default=None)
    parser.add_argument("--qlora", action="store_true", default=False)
    parser.add_argument("--freeze-vision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_EXPORT_ROOT / "train.jsonl")
    parser.add_argument("--val-path", type=Path, default=DEFAULT_EXPORT_ROOT / "val.jsonl")
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--text-mix-root", nargs="*", default=["captions"])
    parser.add_argument("--prompt-text", default="A caption that describes this materials science image is:\n")
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--num-warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--peft-model", type=Path, default=None)
    parser.add_argument("--report-to", default="none")
    args = parser.parse_args()

    output_dir = args.expdir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        fp16=False,
        bf16=True,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=True,
        num_train_epochs=args.num_epochs,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=10,
        eval_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.num_warmup_steps,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=output_dir,
        run_name=args.run_name,
        report_to=args.report_to,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        optim="adamw_torch_fused",
        group_by_length=False,
        max_steps=args.max_steps,
    )

    model, tokenizer, processor = setup_model(args)
    train_dataset, val_dataset = setup_datasets(args, processor)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=LlamaVisionDataCollator(tokenizer),
        callbacks=[LatestCheckpointCallback()],
    )
    trainer.train()


if __name__ == "__main__":
    main()
