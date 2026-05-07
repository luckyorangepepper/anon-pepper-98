from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None


DEFAULT_DATASET = "luckyorangepepper/anon-pepper-72"
IMAGE_MARKER = "{image}"


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


def safe_stem(value: str) -> str:
    cleaned = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "example"


def save_image(row: dict[str, Any], qid: str, image_dir: Path | None) -> str:
    existing = first_present(row, ["image_path", "image_filename", "path"])
    if existing:
        return existing
    if image_dir is None:
        return ""

    image = row.get("image")
    if image is None:
        return ""
    if isinstance(image, dict) and image.get("path"):
        return str(image["path"])
    if not hasattr(image, "save"):
        return ""

    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{safe_stem(qid)}.png"
    image.save(image_path)
    return str(image_path)


def normalize_row(row: dict[str, Any], index: int, image_dir: Path | None) -> dict[str, Any]:
    qid = row_identifier(row, index)
    question = first_present(row, ["question", "prompt", "input_prompt"])
    answer = first_present(
        row,
        ["answer", "ground_truth_answer", "reference_answer", "reference_caption", "caption"],
    )
    image_path = save_image(row, qid, image_dir)
    task_type = first_present(row, ["type", "task_type"])
    if not task_type:
        task_type = "vision" if image_path or IMAGE_MARKER in question or row.get("image") is not None else "text"

    output = {
        "qid": qid,
        "type": task_type,
        "kind": first_present(row, ["kind", "question_type", "modality"]),
        "question": question,
        "answer": answer,
    }
    if image_path:
        output["image_path"] = image_path
    if row.get("paper_id") is not None:
        output["paper_id"] = str(row["paper_id"])
    if row.get("image_id") is not None:
        output["image_id"] = str(row["image_id"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export anonymized MATRIX finetune split to JSONL.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if load_dataset is None:
        raise RuntimeError("Install `datasets` to export the finetune dataset.")

    if args.hf_config:
        dataset = load_dataset(args.dataset, args.hf_config, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            if args.max_samples is not None and count >= args.max_samples:
                break
            normalized = normalize_row(dict(row), index, args.image_dir)
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} rows to {args.output}")
    if args.image_dir:
        print(f"Saved images under {args.image_dir}")


if __name__ == "__main__":
    main()
