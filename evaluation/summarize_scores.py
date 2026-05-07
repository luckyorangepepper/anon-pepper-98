from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    return list(raw.values())


def row_score(row: dict[str, Any]) -> float | None:
    score = row.get("evaluation", {}).get("score")
    if score is None:
        score = row.get("llm_score")
    if score is None:
        score = row.get("score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def print_group(label: str, rows: list[dict[str, Any]]) -> None:
    scores = [score for row in rows if (score := row_score(row)) is not None]
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"{label}\tn={len(scores)}\tavg={avg:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Opus judge scores.")
    parser.add_argument("--scores-file", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.scores_file)
    print_group("ALL", rows)

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("type", "")), str(row.get("kind", "")), str(row.get("rubric", "")))].append(row)
    for (task_type, kind, rubric), group_rows in sorted(groups.items()):
        print_group(f"{task_type}/{kind}/{rubric}", group_rows)


if __name__ == "__main__":
    main()
