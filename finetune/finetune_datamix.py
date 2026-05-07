import os
import argparse
import json
import math
from collections import Counter
from typing import Any, Dict, List, Optional
import torch
from pathlib import Path
from PIL import Image

# Avoid tokenizers parallelism before forking to silence warnings and avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataclasses import dataclass
import transformers
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)
from torch.utils.data import ConcatDataset, Dataset
from transformers import TrainerCallback
import shutil

IGNORE_INDEX = -100
MAX_LENGTH = 3072
MAX_IMAGE_TOKENS = 2500
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_EXPORT_ROOT = Path("finetune/exports")
DEFAULT_IMAGE_ROOT = DEFAULT_EXPORT_ROOT / "images"
DEFAULT_OUTPUT_DIR = Path("finetune/runs")

# Set to None to include all kinds, or a list to filter specific kinds
ALLOWED_KINDS: Optional[List[str]] = ["EDS", "SEM-BSE", "SEM-SE", "TGA", "XRD", "other"]
# ALLOWED_KINDS: Optional[List[str]] = ["EDS", "SEM-BSE", "SEM-SE", "TGA", "XRD"]


def _first_present(record: Dict[str, Any], keys: List[str]) -> str:
    """Return the first non-empty value from a JSONL record as a string."""
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return str(value)
    return ""


def _is_image_record(record: Dict[str, Any]) -> bool:
    question = _first_present(record, ["question", "prompt", "instruction"])
    task_type = _first_present(record, ["type", "task_type"]).lower()
    return (
        task_type in {"vision", "image", "caption"}
        or bool(_first_present(record, ["image_path", "image_filename", "path", "image_id"]))
        or record.get("image") is not None
        or "{image}" in question
        or "<image>" in question
        or "<image_placeholder>" in question
    )


class LatestCheckpointCallback(TrainerCallback):
    """Maintains a 'latest' symlink pointing to the most recent checkpoint."""

    def on_save(self, args, state, control, **kwargs):
        # Only create symlink on main process to avoid race condition in distributed training
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
            pass  # Another process already created the symlink


def _swap_variant(model_id: str, to_variant: str) -> str:
    """Swap model variant (Base/Instruct/Reasoning) in model ID string."""
    for v in ("Base", "Instruct", "Reasoning"):
        model_id = model_id.replace(f"-{v}-", f"-{to_variant}-")
    return model_id


def resolve_weights_and_processor_repo(
    model_name: str,
    *,
    finetune_on_base: bool,
    processor_preference: str = "instruct",
) -> tuple[str, str]:
    """
    Returns (weights_repo, processor_repo).

    - If finetune_on_base=True, weights_repo is forced to Base (even if user passed Instruct).
    - processor_preference="instruct" uses Instruct repo for processor/chat_template.
      processor_preference="match" uses the same repo as weights.
    """
    weights_repo = model_name
    if finetune_on_base:
        weights_repo = _swap_variant(model_name, "Base")

    if processor_preference == "match":
        processor_repo = weights_repo
    else:
        processor_repo = _swap_variant(weights_repo, "Instruct")

    return weights_repo, processor_repo


class ImageCaptionJsonlDataset(Dataset):
    """Dataset for image description SFT using JSONL records.

    Supports both training and inference modes:
    - training: returns full conversation with labels for supervised learning
    - inference: returns only the user prompt ready for generation
    """

    def __init__(
        self,
        jsonl_path: Path,
        processor: AutoProcessor,
        images_root: Optional[Path] = None,
        image_token_index: Optional[int] = None,
        mode: str = "training",
        prompt_style: str = "assistant",
    ):
        if mode not in ("training", "inference"):
            raise ValueError(f"mode must be 'training' or 'inference', got {mode}")
        self.mode = mode
        self.processor = processor
        self.images_root = images_root
        self.prompt_style = prompt_style
        self.prompt_text = self._resolve_prompt_text()
        if image_token_index is None:
            raise RuntimeError("image_token_index must be provided from model.config.image_token_index")
        self.image_token_index = image_token_index
        self.samples: List[Dict] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("deleted") or record.get("empty_caption") or record.get("empty_description"):
                    continue

                # Filter by kind if ALLOWED_KINDS is set
                if ALLOWED_KINDS is not None:
                    kind = record.get("kind")
                    if kind not in ALLOWED_KINDS:
                        continue

                description = _first_present(record, ["description", "answer", "caption", "reference_caption"])
                image_ref = _first_present(record, ["image_path", "image_filename", "path", "image_id"])
                paper_id = _first_present(record, ["paper_id"])
                if not description or not image_ref:
                    continue

                image_path = self._resolve_image_path(paper_id, image_ref)
                if image_path is None:
                    print(f"Warning: missing image for {jsonl_path.name} line {line_number}; skipping.")
                    continue

                self.samples.append(
                    {
                        "image_path": image_path,
                        "description": description,
                        "kind": record.get("kind"),
                        "context": record.get("context", ""),
                    }
                )

        if not self.samples:
            raise ValueError(f"No usable samples found in {jsonl_path}")

        # Log kind statistics
        kind_counts = Counter(s["kind"] for s in self.samples)
        kind_str = ", ".join(f"{k}: {v}" for k, v in sorted(kind_counts.items(), key=lambda x: -x[1]))
        print(f"[{jsonl_path.name}] Kinds: {kind_str}")
        if ALLOWED_KINDS is not None:
            print(f"[{jsonl_path.name}] Filtering to kinds: {ALLOWED_KINDS}")

        # Log context statistics
        samples_with_context = sum(1 for s in self.samples if s["context"])
        context_pct = 100 * samples_with_context / len(self.samples) if self.samples else 0
        print(f"[{jsonl_path.name}] Loaded {len(self.samples)} samples, {samples_with_context} ({context_pct:.1f}%) with context")

    def __len__(self):
        return len(self.samples)
        # return 10

    def _resolve_image_path(self, paper_id: str, image_ref: str) -> Optional[str]:
        """Resolve image path from an exported path or images_root layout."""
        image_path = Path(image_ref)
        has_extension = image_path.suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp")

        candidates: List[Path] = []

        def add_candidate(path: Path) -> None:
            candidates.append(path)
            if not has_extension:
                candidates.append(path.with_suffix(".png"))

        # Exported datasets may already contain a usable relative or absolute path.
        add_candidate(image_path)

        # Layouts rooted at images_root.
        if self.images_root:
            images_root = Path(self.images_root)
            add_candidate(images_root / image_path)
            if paper_id:
                add_candidate(images_root / paper_id / image_ref)

        # Absolute path fallback
        if image_path.is_absolute():
            add_candidate(image_path)

        for cand in candidates:
            if cand.exists():
                return str(cand)

        return None

    def _ensure_valid_image_size(self, image: Image.Image) -> Image.Image:
        """Resize image to fit token budget while meeting minimum requirements."""
        processor = getattr(self.processor, "image_processor", None)
        patch_size = getattr(processor, "patch_size", 14)
        min_required = max(1, patch_size * 2)
        # tokens ≈ (w/patch) * (h/patch), so max_pixels ≈ MAX_IMAGE_TOKENS * patch^2
        max_pixels = MAX_IMAGE_TOKENS * patch_size * patch_size

        width, height = image.size

        # Downscale if pixel area exceeds budget
        if width * height > max_pixels:
            scale = math.sqrt(max_pixels / (width * height))
            width = int(width * scale)
            height = int(height * scale)
            image = image.resize((width, height), resample=Image.Resampling.BICUBIC)

        # Upscale if too small
        if min(width, height) < min_required:
            scale = min_required / float(min(width, height))
            new_width = int(math.ceil(width * scale))
            new_height = int(math.ceil(height * scale))
            image = image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)

        return image

    def _assert_has_image_tokens(self, input_ids: torch.Tensor, image_token_index: int) -> None:
        """Assert that image tokens exist in input_ids to prevent silent failures."""
        n = (input_ids == image_token_index).sum().item()
        if n == 0:
            raise RuntimeError(
                f"No image tokens found in input_ids (expected token id {image_token_index}). "
                "Your chat template/prompt is not inserting image placeholders."
            )

    def _resolve_prompt_text(self) -> str:
        if self.prompt_style == "base":
            return "A caption that describes this materials science image is:"
        return "What is a caption that is suitable for this image?"

    def _render_mm_prompt(self) -> str:
        """Render the image token placeholder for the prompt."""
        return "<|vision_start|><|image_pad|><|vision_end|>"

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        description = str(sample["description"])
        context = sample["context"]
        with Image.open(image_path) as im:
            image = im.convert("RGB")

        image = self._ensure_valid_image_size(image)

        # Build prompt text with optional context
        if context:
            full_prompt = f"{context}\n\n{self.prompt_text}"
        else:
            full_prompt = self.prompt_text

        # Build chat messages
        input_content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": full_prompt},
        ]

        if self.mode == "inference":
            # Inference mode: return only user prompt for generation
            user_messages = [{"role": "user", "content": input_content}]

            # Apply chat template
            prompt = None
            if hasattr(self.processor, 'apply_chat_template'):
                prompt = self.processor.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            # Fallback
            if not prompt or len(prompt.strip()) == 0:
                image_tokens = self._render_mm_prompt()
                prompt = f"<|im_start|>user\n{image_tokens}{full_prompt}<|im_end|>\n<|im_start|>assistant\n"

            inputs = self.processor(text=[prompt], images=[image], return_tensors="pt", padding=True)

            # Return dict without labels (ready for generation)
            # Keep batch dimension for model.generate()
            return dict(inputs)

        # Training mode: return full conversation with labels
        messages = [
            {"role": "user", "content": input_content},
            {"role": "assistant", "content": [{"type": "text", "text": description}]},
        ]

        # Format conversation - use chat template if available, otherwise manual formatting
        rendered = None
        if hasattr(self.processor, 'apply_chat_template'):
            rendered = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        # Fallback if chat template doesn't work or returns empty
        if not rendered or len(rendered.strip()) == 0:
            image_tokens = self._render_mm_prompt()
            eos_token = getattr(self.processor.tokenizer, "eos_token", "") or ""
            rendered = f"<|im_start|>user\n{image_tokens}{full_prompt}<|im_end|>\n<|im_start|>assistant\n{description}<|im_end|>{eos_token}"

        full = self.processor(text=[rendered], images=[image], return_tensors="pt")

        self._assert_has_image_tokens(full["input_ids"][0], self.image_token_index)

        input_ids = full["input_ids"][0]
        labels = input_ids.clone()
        attention_mask = full["attention_mask"][0]

        # Mask prompt tokens - only train on assistant response
        user_messages = [{"role": "user", "content": input_content}]

        user_rendered = None
        if hasattr(self.processor, 'apply_chat_template'):
            user_rendered = self.processor.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback if chat template doesn't work or returns empty
        if not user_rendered or len(user_rendered.strip()) == 0:
            image_tokens = self._render_mm_prompt()
            user_rendered = f"<|im_start|>user\n{image_tokens}{full_prompt}<|im_end|>\n<|im_start|>assistant\n"

        user = self.processor(text=[user_rendered], images=[image], return_tensors="pt")
        user_len = min(user["input_ids"].shape[-1], labels.shape[-1])
        labels[:user_len] = IGNORE_INDEX

        # Ensure EOS token is NOT masked so model learns to generate it
        eos_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            eos_positions = (input_ids == eos_id).nonzero(as_tuple=False).flatten().tolist()
            for pos in eos_positions:
                if pos >= user_len:
                    labels[pos] = eos_id

        if input_ids.shape[-1] > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]

        example: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        if "pixel_values" in full:
            # Keep full patch sequence so batches can be concatenated later.
            example["pixel_values"] = full["pixel_values"]
        if "image_sizes" in full:
            example["image_sizes"] = full["image_sizes"]
        if "image_grid_thw" in full:
            example["image_grid_thw"] = full["image_grid_thw"]

        return example


class TextJsonlDataset(Dataset):
    """Dataset for text-only SFT JSONL records (e.g., QA pairs)."""

    def __init__(
        self,
        jsonl_path: Path,
        processor: AutoProcessor,
        question_key: str = "question",
        answer_key: str = "answer",
    ):
        self.processor = processor
        self.question_key = question_key
        self.answer_key = answer_key
        self.samples: List[Dict[str, str]] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if _is_image_record(record):
                    continue
                question = record.get(self.question_key) or record.get("prompt") or record.get("instruction")
                answer = record.get(self.answer_key) or record.get("response") or record.get("output")
                if not question or not answer:
                    continue
                self.samples.append({"question": str(question), "answer": str(answer)})

        if not self.samples:
            raise ValueError(f"No usable samples found in {jsonl_path}")

    def __len__(self):
        return len(self.samples)
        # return 10

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        question = sample["question"]
        answer = sample["answer"]

        # Build chat messages per Qwen2-VL chat template (matching Qwen2VLSFTDataset)
        input_content = [{"type": "text", "text": question}]
        messages = [
            {"role": "user", "content": input_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]

        # Format conversation - use chat template if available, otherwise manual formatting
        rendered = None
        if hasattr(self.processor, 'apply_chat_template'):
            rendered = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        # Fallback if chat template doesn't work or returns empty
        if not rendered or len(rendered.strip()) == 0:
            eos_token = getattr(self.processor.tokenizer, "eos_token", "") or ""
            rendered = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>{eos_token}"

        model_inputs = self.processor(text=[rendered], return_tensors="pt")
        input_ids = model_inputs["input_ids"][0]
        labels = input_ids.clone()
        attention_mask = model_inputs["attention_mask"][0]

        # Mask prompt tokens - only train on assistant response
        user_messages = [{"role": "user", "content": input_content}]

        user_rendered = None
        if hasattr(self.processor, 'apply_chat_template'):
            user_rendered = self.processor.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback if chat template doesn't work or returns empty
        if not user_rendered or len(user_rendered.strip()) == 0:
            user_rendered = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

        user_inputs = self.processor(text=[user_rendered], return_tensors="pt")
        user_len = min(user_inputs["input_ids"].shape[-1], labels.shape[-1])
        labels[:user_len] = IGNORE_INDEX

        # Ensure EOS token is NOT masked so model learns to generate it
        eos_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            eos_positions = (input_ids == eos_id).nonzero(as_tuple=False).flatten().tolist()
            for pos in eos_positions:
                if pos >= user_len:
                    labels[pos] = eos_id

        if input_ids.shape[-1] > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class ScienceQADataset(Dataset):
    """Dataset for ScienceQA multiple-choice questions with optional images."""

    def __init__(
        self,
        jsonl_path: Path,
        processor: AutoProcessor,
        images_root: Optional[Path] = None,
        image_token_index: Optional[int] = None,
        mode: str = "training",
    ):
        if mode not in ("training", "inference"):
            raise ValueError(f"mode must be 'training' or 'inference', got {mode}")
        self.mode = mode
        self.processor = processor
        self.images_root = images_root
        self.image_token_index = image_token_index
        self.samples: List[Dict] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                # Required fields
                question = record.get("question")
                choices = record.get("choices")
                answer_text = record.get("answer_text")

                if not question or not choices or not answer_text:
                    continue

                self.samples.append({
                    "question": question,
                    "choices": choices,
                    "answer_text": answer_text,
                    "image_id": record.get("image_id"),
                    "has_image": record.get("has_image", False),
                    "context": record.get("context", ""),
                })

        if not self.samples:
            raise ValueError(f"No usable samples found in {jsonl_path}")

        # Log statistics
        num_with_image = sum(1 for s in self.samples if s["has_image"])
        num_without_image = len(self.samples) - num_with_image
        print(f"[{jsonl_path.name}] Loaded {len(self.samples)} samples: {num_with_image} with image, {num_without_image} text-only")

    def __len__(self):
        return len(self.samples)

    def _ensure_valid_image_size(self, image: Image.Image) -> Image.Image:
        """Resize image to fit token budget while meeting minimum requirements."""
        processor = getattr(self.processor, "image_processor", None)
        patch_size = getattr(processor, "patch_size", 14)
        min_required = max(1, patch_size * 2)
        max_pixels = MAX_IMAGE_TOKENS * patch_size * patch_size

        width, height = image.size

        if width * height > max_pixels:
            scale = math.sqrt(max_pixels / (width * height))
            width = int(width * scale)
            height = int(height * scale)
            image = image.resize((width, height), resample=Image.Resampling.BICUBIC)

        if min(width, height) < min_required:
            scale = min_required / float(min(width, height))
            new_width = int(math.ceil(width * scale))
            new_height = int(math.ceil(height * scale))
            image = image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)

        return image

    def _format_prompt(self, sample: Dict) -> str:
        """Format the question and choices into a prompt."""
        choice_labels = ["A", "B", "C", "D", "E"]
        choices_text = "\n".join(
            f"{choice_labels[i]}. {choice}" for i, choice in enumerate(sample["choices"])
        )

        prompt = f"Question: {sample['question']}\n\n{choices_text}\n\nAnswer with the letter of the correct choice."

        # Prepend context (hint) if available
        context = sample.get("context", "")
        if context:
            prompt = f"{context}\n\n{prompt}"

        return prompt

    def _render_mm_prompt(self) -> str:
        """Render the image token placeholder for the prompt."""
        return "<|vision_start|><|image_pad|><|vision_end|>"

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        has_image = sample["has_image"] and sample["image_id"]
        prompt = self._format_prompt(sample)
        answer = sample["answer_text"]  # e.g., "B"

        # Load image if present
        image = None
        if has_image and self.images_root:
            image_path = self.images_root / sample["image_id"]
            if image_path.exists():
                with Image.open(image_path) as im:
                    image = im.convert("RGB")
                image = self._ensure_valid_image_size(image)
            else:
                has_image = False

        # Build chat messages
        if has_image and image is not None:
            input_content = [
                {"type": "image", "image": str(self.images_root / sample["image_id"])},
                {"type": "text", "text": prompt},
            ]
        else:
            input_content = [{"type": "text", "text": prompt}]

        if self.mode == "inference":
            user_messages = [{"role": "user", "content": input_content}]

            rendered = None
            if hasattr(self.processor, 'apply_chat_template'):
                rendered = self.processor.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            if not rendered or len(rendered.strip()) == 0:
                if has_image:
                    image_tokens = self._render_mm_prompt()
                    rendered = f"<|im_start|>user\n{image_tokens}{prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    rendered = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

            if has_image and image is not None:
                inputs = self.processor(text=[rendered], images=[image], return_tensors="pt", padding=True)
            else:
                inputs = self.processor(text=[rendered], return_tensors="pt", padding=True)

            return dict(inputs)

        # Training mode: return full conversation with labels
        messages = [
            {"role": "user", "content": input_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]

        rendered = None
        if hasattr(self.processor, 'apply_chat_template'):
            rendered = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        if not rendered or len(rendered.strip()) == 0:
            eos_token = getattr(self.processor.tokenizer, "eos_token", "") or ""
            if has_image:
                image_tokens = self._render_mm_prompt()
                rendered = f"<|im_start|>user\n{image_tokens}{prompt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>{eos_token}"
            else:
                rendered = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>{eos_token}"

        if has_image and image is not None:
            full = self.processor(text=[rendered], images=[image], return_tensors="pt")
        else:
            full = self.processor(text=[rendered], return_tensors="pt")

        input_ids = full["input_ids"][0]
        labels = input_ids.clone()
        attention_mask = full["attention_mask"][0]

        # Mask prompt tokens - only train on assistant response
        user_messages = [{"role": "user", "content": input_content}]

        user_rendered = None
        if hasattr(self.processor, 'apply_chat_template'):
            user_rendered = self.processor.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        if not user_rendered or len(user_rendered.strip()) == 0:
            if has_image:
                image_tokens = self._render_mm_prompt()
                user_rendered = f"<|im_start|>user\n{image_tokens}{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                user_rendered = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        if has_image and image is not None:
            user_inputs = self.processor(text=[user_rendered], images=[image], return_tensors="pt")
        else:
            user_inputs = self.processor(text=[user_rendered], return_tensors="pt")

        user_len = min(user_inputs["input_ids"].shape[-1], labels.shape[-1])
        labels[:user_len] = IGNORE_INDEX

        # Ensure EOS token is NOT masked
        eos_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            eos_positions = (input_ids == eos_id).nonzero(as_tuple=False).flatten().tolist()
            for pos in eos_positions:
                if pos >= user_len:
                    labels[pos] = eos_id

        if input_ids.shape[-1] > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]

        example = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # Include vision inputs if present
        if "pixel_values" in full:
            example["pixel_values"] = full["pixel_values"]
        if "image_sizes" in full:
            example["image_sizes"] = full["image_sizes"]
        if "image_grid_thw" in full:
            example["image_grid_thw"] = full["image_grid_thw"]

        return example


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids = [inst["input_ids"].clone() for inst in instances]
        labels = [inst["labels"].clone() for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Pad to multiple of 8 for Tensor Core efficiency
        def pad8(t, val):
            r = t.size(1) % 8
            return t if r == 0 else torch.nn.functional.pad(t, (0, 8 - r), value=val)

        input_ids = pad8(input_ids, self.tokenizer.pad_token_id)
        labels = pad8(labels, IGNORE_INDEX)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

        # Collect vision inputs
        if any("pixel_values" in inst for inst in instances):
            pv = [inst["pixel_values"] for inst in instances if "pixel_values" in inst]
            if pv:
                fused = torch.cat(pv, dim=0)
                batch["pixel_values"] = fused.to(torch.bfloat16) if fused.dtype == torch.float32 else fused

        if any("image_sizes" in inst for inst in instances):
            sizes = [inst["image_sizes"] for inst in instances if "image_sizes" in inst]
            if sizes:
                batch["image_sizes"] = torch.cat(sizes, dim=0)

        if any("image_grid_thw" in inst for inst in instances):
            grids = []
            for inst in instances:
                g = inst.get("image_grid_thw")
                if g is not None:
                    grids.append(g if isinstance(g, torch.Tensor) else torch.tensor(g, dtype=torch.long))
            if grids:
                batch["image_grid_thw"] = torch.cat([g.view(-1, 3) for g in grids], dim=0)

        return batch


def _find_jsonl_files(root: Path, split: str) -> List[Path]:
    """Find {split}.jsonl files in a directory."""
    if not root.exists():
        return []
    if root.is_file() and root.suffix == ".jsonl":
        return [root]
    results = []
    direct = root / f"{split}.jsonl"
    if direct.exists():
        results.append(direct)
    results.extend(sorted(root.rglob(f"{split}.jsonl")))
    return list(dict.fromkeys(results))  # dedupe preserving order


def setup_datasets(args, processor, model_config):
    # Get image token index
    image_token_index = getattr(model_config, "image_token_index", None)
    if image_token_index is None and hasattr(processor, "tokenizer"):
        image_token_index = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_index is None or image_token_index < 0:
        raise RuntimeError(f"Could not determine image_token_index for {type(model_config)}")

    # Create caption datasets (always use assistant style for training)
    def make_caption_ds(path):
        return ImageCaptionJsonlDataset(
            jsonl_path=path, processor=processor, images_root=args.images_root,
            image_token_index=image_token_index, prompt_style="assistant",
        )

    # Create ScienceQA datasets if paths are provided
    def make_scienceqa_ds(path, images_root):
        return ScienceQADataset(
            jsonl_path=path, processor=processor, images_root=images_root,
            image_token_index=image_token_index, mode="training",
        )

    train_datasets, val_datasets = [], []
    caption_added = False
    scienceqa_added = False

    entries = args.text_mix_root or []
    if not entries:
        raise ValueError("--text-mix-root must include at least one dataset source.")

    for entry in entries:
        if isinstance(entry, str) and entry.lower() == "captions":
            if not caption_added:
                train_datasets.append(make_caption_ds(args.train_path))
                val_datasets.append(make_caption_ds(args.val_path))
                caption_added = True
        elif isinstance(entry, str) and entry.lower() == "scienceqa":
            if not scienceqa_added and args.scienceqa_train_path and args.scienceqa_val_path:
                train_datasets.append(make_scienceqa_ds(args.scienceqa_train_path, args.scienceqa_images_root))
                val_datasets.append(make_scienceqa_ds(args.scienceqa_val_path, args.scienceqa_images_root))
                scienceqa_added = True
        else:
            root = Path(entry)
            for path in _find_jsonl_files(root, "train"):
                train_datasets.append(TextJsonlDataset(path, processor))
            for path in _find_jsonl_files(root, "val"):
                val_datasets.append(TextJsonlDataset(path, processor))

    if not train_datasets or not val_datasets:
        raise ValueError("No train/val datasets were created from --text-mix-root.")

    train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    val_dataset = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)

    print(f"[datamix] {len(train_datasets)} datasets, {len(train_dataset)} train, {len(val_dataset)} val samples")
    return {"train": train_dataset, "val": val_dataset}


def resolve_deepspeed_config(args):
    """Build a DeepSpeed ZeRO-2 config for Trainer."""
    try:
        import deepspeed  # type: ignore  # noqa: F401
    except Exception:
        print("Warning: DeepSpeed not installed; continuing without it.")
        return None

    try:
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        world_size = 1

    total_train_batch_size = args.batch_size * args.grad_accum * world_size
    return {
        "train_batch_size": total_train_batch_size,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 50,
        "wall_clock_breakdown": False,
        "bf16": {"enabled": True},
    }


def setup_training_args(args):
    output_dir = args.expdir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    deepspeed_config = resolve_deepspeed_config(args)
    optim_name = "adamw_torch" if deepspeed_config else "paged_adamw_8bit"

    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    training_args = TrainingArguments(
        fsdp=None,  # Disable FSDP due to integer tensor issues
        fp16=False,  # Disable FP16 to prevent overflow
        bf16=True,   # Force BF16 for better stability and larger dynamic range
        gradient_checkpointing=True,  # Enable for memory efficiency
        ddp_find_unused_parameters=False,
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
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=output_dir,
        run_name=args.run_name,
        report_to=getattr(args, "report_to", "none"),
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        remove_unused_columns=False,
        optim=optim_name,
        group_by_length=False,
        deepspeed=deepspeed_config,
        max_steps=args.max_steps,
    )
    return training_args


def prepare_tokenizer_and_model(tokenizer, model) -> None:
    """Prepare tokenizer and model: set properties, add special tokens, resize embeddings."""
    tokenizer.model_max_length = MAX_LENGTH
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens_dict = {}
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_emb = model.get_input_embeddings().weight.data
        output_emb = model.get_output_embeddings().weight.data
        if tokenizer.unk_token_id is not None:
            init_emb = input_emb[tokenizer.unk_token_id].unsqueeze(0)
            out_init = output_emb[tokenizer.unk_token_id].unsqueeze(0)
        else:
            init_emb = input_emb[:-num_new_tokens].mean(dim=0, keepdim=True)
            out_init = output_emb[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_emb[-num_new_tokens:] = init_emb
        output_emb[-num_new_tokens:] = out_init

    true_vocab_size = model.get_output_embeddings().weight.shape[0]
    model.config.vocab_size = true_vocab_size
    if hasattr(model.config, "text_config"):
        model.config.text_config.vocab_size = true_vocab_size


def setup_model(args):
    weights_repo, processor_repo = resolve_weights_and_processor_repo(
        args.model_name,
        finetune_on_base=args.finetune_on_base,
        processor_preference=args.processor_preference,
    )
    tokenizer_string = args.tokenizer_name or processor_repo

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False,
    ) if args.fp8 else None

    model = AutoModelForImageTextToText.from_pretrained(
        weights_repo,
        quantization_config=quantization_config,
        device_map=None,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    # Fix: make Mistral3Config expose vocab_size so PEFT save logic doesn't crash
    Cfg = model.config.__class__
    if not hasattr(Cfg, "vocab_size"):
        def _get_vs(self):
            tc = getattr(self, "text_config", None)
            return getattr(tc, "vocab_size", None) if tc else getattr(self, "_vocab_size", None)
        def _set_vs(self, v):
            tc = getattr(self, "text_config", None)
            if tc: tc.vocab_size = v
            else: self._vocab_size = v
        Cfg.vocab_size = property(_get_vs, _set_vs)

    if args.fp8:
        model = prepare_model_for_kbit_training(model)

    processor = AutoProcessor.from_pretrained(tokenizer_string, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(weights_repo, trust_remote_code=True, use_fast=True)
    processor.tokenizer = tokenizer

    # Apply LoRA (skip for inference when apply_lora=False)
    if getattr(args, "apply_lora", True):
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
        if torch.cuda.is_available():
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    prepare_tokenizer_and_model(tokenizer, model)

    if args.peft_model is not None:
        model = PeftModel.from_pretrained(model, args.peft_model, device_map="auto")

    return model, tokenizer, processor


def setup_trainer(args):
    training_args = setup_training_args(args)
    model, tokenizer, processor = setup_model(args)

    datasets = setup_datasets(args, processor, model.config)

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=data_collator,
        callbacks=[LatestCheckpointCallback()],
    )
    return trainer


def main(args):
    trainer = setup_trainer(args)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--expdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="Qwen/Qwen2-VL-7B")
    parser.add_argument("--fp8", "--load-in-8bit", dest="fp8", action="store_true", default=False)
    parser.add_argument("--no-fp8", "--no-load-in-8bit", dest="fp8", action="store_false")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_EXPORT_ROOT / "train.jsonl")
    parser.add_argument("--val-path", type=Path, default=DEFAULT_EXPORT_ROOT / "val.jsonl")
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument(
        "--text-mix-root",
        nargs="*",
        default=["captions"],
        help="Dataset sources to mix: use 'captions', 'scienceqa', or JSONL directories.",
    )
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--num-warmup-steps", type=int, default=100)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-freq", default=1000, type=int)
    parser.add_argument("--save-freq", default=1000, type=int)
    parser.add_argument("--peft-model", type=Path, default=None)
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument("--finetune-on-base", action="store_true", default=True)
    parser.add_argument("--processor-preference", type=str, default="match", choices=["instruct", "match"])
    # ScienceQA dataset paths
    parser.add_argument("--scienceqa-train-path", type=Path, default=None)
    parser.add_argument("--scienceqa-val-path", type=Path, default=None)
    parser.add_argument("--scienceqa-images-root", type=Path, default=None)
    args = parser.parse_args()

    main(args)
