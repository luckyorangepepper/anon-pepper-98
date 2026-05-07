from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None


DEFAULT_DATASET = "luckyorangepepper/anon-pepper-72"
DEFAULT_JUDGE_MODEL = "claude-opus-4-5"
ALLOWED_SCORES = [0.0, 0.25, 0.5, 0.75, 1.0]
RUBRIC_DIR = Path(__file__).resolve().parent / "rubrics"
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
RUBRIC_ALIASES = {
    "text_qa": "text_qa",
    "text": "text_qa",
    "qa": "text_qa",
    "hypothesis_generation": "hypothesis_generation",
    "hypothesis": "hypothesis_generation",
    "hypo": "hypothesis_generation",
    "image_caption": "image_caption",
    "caption": "image_caption",
    "image": "image_caption",
    "vision": "image_caption",
}


@dataclass
class Entry:
    original_index: int
    qid: str
    paper_id: str
    image_id: str
    image_path: str
    task_type: str
    kind: str
    question: str
    answer: str
    model_answer: str
    rubric_name: str


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


def load_hf_rows(dataset_name: str, split: str, config_name: str | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install `datasets` to join predictions with the Hugging Face dataset.") from exc

    if config_name:
        dataset = load_dataset(dataset_name, config_name, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
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


def row_identifiers(row: dict[str, Any], fallback_index: int) -> list[str]:
    identifiers = []
    for key in ["qid", "id", "image_id", "paper_id"]:
        value = first_present(row, [key])
        if value and value not in identifiers:
            identifiers.append(value)

    paper_id = first_present(row, ["paper_id"])
    image_id = first_present(row, ["image_id"])
    if paper_id and image_id:
        combined = f"{paper_id}:{image_id}"
        if combined not in identifiers:
            identifiers.append(combined)

    if not identifiers:
        identifiers.append(str(fallback_index))
    return identifiers


def merge_with_dataset(
    prediction_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dataset_by_id = {}
    for index, row in enumerate(dataset_rows):
        for identifier in row_identifiers(row, index):
            dataset_by_id.setdefault(identifier, row)

    merged_rows = []
    missing = 0
    for index, prediction in enumerate(prediction_rows):
        reference = None
        for key in row_identifiers(prediction, index):
            reference = dataset_by_id.get(key)
            if reference is not None:
                break
        if reference is None:
            missing += 1
            merged_rows.append(prediction)
            continue
        merged = dict(reference)
        merged.update(prediction)
        merged_rows.append(merged)

    if missing:
        print(f"Warning: {missing}/{len(prediction_rows)} prediction rows did not match the dataset by id.")
    return merged_rows


def choose_auto_rubric(row: dict[str, Any]) -> str:
    task_type = str(row.get("type") or row.get("task_type") or "").lower()
    kind = str(row.get("kind") or row.get("question_type") or row.get("modality") or "").lower()
    combined = f"{task_type} {kind}"

    if "hypothesis" in combined or "hypo" in combined:
        return "hypothesis_generation"
    if row.get("reference_caption") or row.get("predicted_caption") or row.get("image_path") or task_type == "vision":
        return "image_caption"
    if any(label in combined for label in IMAGE_CAPTION_KINDS):
        return "image_caption"
    return "text_qa"


def normalize_rubric_arg(rubric: str, row: dict[str, Any]) -> str:
    if rubric == "auto":
        return choose_auto_rubric(row)
    if rubric in RUBRIC_ALIASES:
        return RUBRIC_ALIASES[rubric]
    path = Path(rubric)
    if path.exists():
        return str(path)
    raise ValueError(
        "--rubric must be auto, text_qa, hypothesis_generation, image_caption, or a path to a rubric file."
    )


def load_rubric(rubric_name: str) -> str:
    path = Path(rubric_name)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return (RUBRIC_DIR / f"{rubric_name}.md").read_text(encoding="utf-8").strip()


def load_entries(
    path: Path,
    rubric: str,
    max_samples: int | None,
    dataset_name: str | None,
    split: str,
    hf_config: str | None,
) -> list[Entry]:
    rows = read_json_or_jsonl(path)
    if dataset_name:
        rows = merge_with_dataset(rows, load_hf_rows(dataset_name, split, hf_config))
    if max_samples is not None:
        rows = rows[:max_samples]

    entries: list[Entry] = []
    for index, row in enumerate(rows):
        rubric_name = normalize_rubric_arg(rubric, row)
        entries.append(
            Entry(
                original_index=index,
                qid=row_identifier(row, index),
                paper_id=first_present(row, ["paper_id"]),
                image_id=first_present(row, ["image_id"]),
                image_path=first_present(row, ["image_path"]),
                task_type=first_present(row, ["type", "task_type"]),
                kind=first_present(row, ["kind", "question_type", "modality"]),
                question=first_present(row, ["question", "prompt"]),
                answer=first_present(
                    row,
                    ["answer", "ground_truth_answer", "reference_answer", "reference_caption"],
                ),
                model_answer=first_present(
                    row,
                    ["model_answer", "prediction", "output", "response", "generated_answer", "predicted_caption"],
                ),
                rubric_name=rubric_name,
            )
        )
    return entries


def format_batch(rubric_text: str, batch: list[Entry]) -> str:
    if len(batch) == 1 and batch[0].rubric_name == "image_caption":
        item = batch[0]
        return rubric_text.format(
            reference=item.answer.strip(),
            prediction=item.model_answer.strip(),
        )

    rows = []
    for local_index, item in enumerate(batch):
        rows.append(
            f"Item {local_index}\n"
            f"Question: {item.question}\n"
            f"Ground Truth Answer: {item.answer}\n"
            f"Model Answer: {item.model_answer}\n"
        )
    return (
        f"{rubric_text}\n\n"
        "Evaluate the following items.\n\n"
        + "\n".join(rows)
        + '\nReturn JSON list like: [{"index":0,"score":1.0,"reason":"..."}]'
    )


def use_foundry_anthropic() -> bool:
    return os.getenv("CLAUDE_CODE_USE_FOUNDRY") == "1" or bool(
        os.getenv("ANTHROPIC_FOUNDRY_RESOURCE") or os.getenv("ANTHROPIC_FOUNDRY_BASE_URL")
    )


def build_client(api_key: str | None) -> Any:
    if anthropic is None:
        raise ImportError("Install the anthropic package to run the Opus judge.")

    if use_foundry_anthropic():
        api_key = api_key or os.getenv("ANTHROPIC_FOUNDRY_API_KEY")
        if not api_key:
            raise ValueError("Set ANTHROPIC_FOUNDRY_API_KEY or pass --api-key.")
        if not hasattr(anthropic, "AnthropicFoundry"):
            raise ImportError("Upgrade anthropic to use AnthropicFoundry.")
        resource = os.getenv("ANTHROPIC_FOUNDRY_RESOURCE")
        if resource:
            return anthropic.AnthropicFoundry(api_key=api_key, resource=resource)
        base_url = os.getenv("ANTHROPIC_FOUNDRY_BASE_URL")
        if not base_url:
            raise ValueError("Set ANTHROPIC_FOUNDRY_RESOURCE or ANTHROPIC_FOUNDRY_BASE_URL.")
        return anthropic.AnthropicFoundry(api_key=api_key, base_url=base_url.rstrip("/"))

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY or pass --api-key.")
    return anthropic.Anthropic(api_key=api_key)


def extract_text(response: Any) -> str:
    chunks = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", "") == "text":
            chunks.append(block.text)
    return "".join(chunks).strip()


def parse_json_response(content: str) -> list[Any]:
    if not content.strip():
        raise ValueError("Empty judge response.")

    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def canonicalize_score(raw_score: Any) -> float:
    score = float(raw_score)
    return min(ALLOWED_SCORES, key=lambda allowed: abs(allowed - score))


def normalize_scores(parsed: Any, expected_len: int) -> list[dict[str, Any]]:
    if isinstance(parsed, dict):
        parsed = [
            {
                "index": 0,
                "score": parsed.get("score"),
                "reason": parsed.get("reason", parsed.get("explanation", "")),
            }
        ]

    normalized = []
    for fallback_index, item in enumerate(parsed):
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "index": int(item.get("index", fallback_index)),
                "score": canonicalize_score(item.get("score")),
                "reason": str(item.get("reason", ""))[:200],
            }
        )

    if len(normalized) > expected_len:
        normalized = normalized[:expected_len]
    if len(normalized) != expected_len:
        raise ValueError(f"Expected {expected_len} scored items, got {len(normalized)}.")
    return normalized


def score_batch(
    client: Any,
    *,
    judge_model: str,
    rubric_text: str,
    batch: list[Entry],
    max_tokens: int,
) -> list[dict[str, Any]]:
    response = client.messages.create(
        model=judge_model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[{"role": "user", "content": format_batch(rubric_text, batch)}],
    )
    content = extract_text(response)
    return normalize_scores(parse_json_response(content), len(batch))


def score_group(
    client: Any,
    *,
    judge_model: str,
    rubric_text: str,
    entries: list[Entry],
    batch_size: int,
    max_tokens: int,
    max_retries: int,
) -> list[dict[str, Any]]:
    rows = []
    index = 0
    current_batch_size = batch_size
    current_max_tokens = max_tokens
    retries = 0

    while index < len(entries):
        effective_batch_size = 1 if entries[index].rubric_name == "image_caption" else current_batch_size
        batch = entries[index : index + effective_batch_size]
        try:
            parsed = score_batch(
                client,
                judge_model=judge_model,
                rubric_text=rubric_text,
                batch=batch,
                max_tokens=current_max_tokens,
            )
        except Exception as exc:
            retries += 1
            if retries > max_retries:
                raise
            message = str(exc).lower()
            if current_batch_size > 1 and any(token in message for token in ["token", "length", "parse", "json"]):
                current_batch_size = max(1, current_batch_size // 2)
                continue
            if any(token in message for token in ["token", "length"]):
                current_max_tokens = min(16384, current_max_tokens * 2)
                continue
            time.sleep(1)
            continue

        for local_index, evaluation in enumerate(parsed):
            entry = batch[local_index]
            row = {
                "qid": entry.qid,
                "type": entry.task_type,
                "kind": entry.kind,
                "rubric": entry.rubric_name,
                "question": entry.question,
                "answer": entry.answer,
                "model_answer": entry.model_answer,
                "evaluation": evaluation,
                "_original_index": entry.original_index,
            }
            if entry.paper_id:
                row["paper_id"] = entry.paper_id
            if entry.image_id:
                row["image_id"] = entry.image_id
            if entry.image_path:
                row["image_path"] = entry.image_path
            if entry.rubric_name == "image_caption":
                row["reference_caption"] = entry.answer
                row["predicted_caption"] = entry.model_answer
            rows.append(row)
        index += effective_batch_size
        current_batch_size = batch_size
        current_max_tokens = max_tokens
        retries = 0
        print(f"Scored {index}/{len(entries)} for rubric={batch[0].rubric_name}")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    scores = [
        float(row["evaluation"]["score"])
        for row in rows
        if row.get("evaluation", {}).get("score") is not None
    ]
    if not scores:
        return {"n": 0, "avg_score": 0.0}
    return {"n": len(scores), "avg_score": sum(scores) / len(scores)}


def write_outputs(rows: list[dict[str, Any]], output_file: Path, summary_csv: Path | None) -> None:
    rows = sorted(rows, key=lambda row: row.pop("_original_index"))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.suffix.lower() == ".jsonl":
        with output_file.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        output_file.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    if summary_csv:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault((row.get("type", ""), row.get("kind", ""), row.get("rubric", "")), []).append(row)

        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["type", "kind", "rubric", "n", "avg_score"])
            overall = summarize(rows)
            writer.writerow(["ALL", "ALL", "ALL", overall["n"], f"{overall['avg_score']:.6f}"])
            for (task_type, kind, rubric), group_rows in sorted(groups.items()):
                stats = summarize(group_rows)
                writer.writerow([task_type, kind, rubric, stats["n"], f"{stats['avg_score']:.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Score MATRIX predictions with the Opus 4.5 judge.")
    parser.add_argument(
        "--answers-file",
        "--predictions-file",
        dest="answers_file",
        type=Path,
        required=True,
        help="Prediction JSON/JSONL. Rows may include references, or use --dataset to join them in.",
    )
    parser.add_argument("--output-file", type=Path, default=Path("outputs/opus_scores.jsonl"))
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument(
        "--rubric",
        default="auto",
        help="auto, text_qa, hypothesis_generation, image_caption, or rubric file path.",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--dataset",
        default=None,
        help=f"Optional Hugging Face dataset used to attach questions/references by id. Default dataset is {DEFAULT_DATASET}.",
    )
    parser.add_argument("--split", default="test", help="Dataset split when using --dataset.")
    parser.add_argument("--hf-config", default=None, help="Optional Hugging Face dataset config.")
    args = parser.parse_args()

    client = build_client(args.api_key)
    entries = load_entries(
        args.answers_file,
        args.rubric,
        args.max_samples,
        args.dataset,
        args.split,
        args.hf_config,
    )
    if not entries:
        raise ValueError("No entries found.")

    print(f"Dataset: {DEFAULT_DATASET}")
    print(f"Judge: {args.judge_model}")
    rows = []
    for rubric_name in sorted({entry.rubric_name for entry in entries}):
        group = [entry for entry in entries if entry.rubric_name == rubric_name]
        rows.extend(
            score_group(
                client,
                judge_model=args.judge_model,
                rubric_text=load_rubric(rubric_name),
                entries=group,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
            )
        )

    write_outputs(rows, args.output_file, args.summary_csv)
    stats = summarize(rows)
    print(f"Saved {len(rows)} rows to {args.output_file}; average score={stats['avg_score']:.4f}")
    if args.summary_csv:
        print(f"Saved summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
