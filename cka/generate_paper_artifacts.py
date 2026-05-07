from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE_PAIRS = [
    ("base", "text-sft", "Text SFT"),
    ("base", "vision-sft", "Vision SFT"),
    ("base", "full-sft", "Full SFT"),
]
PAIR_COLORS = {"Text SFT": "#2F6BFF", "Vision SFT": "#D97706", "Full SFT": "#168A4A"}


def load_rows(paths: list[Path], family: str) -> list[dict]:
    rows = []
    for path in paths:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"Expected a JSON list in {path}")
        rows.extend(loaded)
    for row in rows:
        row["family"] = family
    return rows


def group_rows(rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["model_a"], row["model_b"])].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda item: item["layer"])
    return grouped


def summarize(rows: list[dict]) -> list[dict]:
    grouped = group_rows(rows)
    output = []
    for family in sorted({row["family"] for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        sources = sorted({row["source"] for row in family_rows})
        for source in sources:
            for model_a, model_b, label in BASE_PAIRS:
                series = grouped.get((source, model_a, model_b), [])
                series = [row for row in series if row["family"] == family]
                if not series:
                    continue
                max_layer = max(row["layer"] for row in series)
                late_start = int(np.ceil(max_layer * 0.75))
                late = [row for row in series if row["layer"] >= late_start]
                max_div = max(series, key=lambda row: 1.0 - row["cka"])
                final = next(row for row in series if row["layer"] == max_layer)
                output.append(
                    {
                        "family": family,
                        "source": source,
                        "comparison": label,
                        "max_divergence_pct": round((1.0 - max_div["cka"]) * 100.0, 3),
                        "max_divergence_layer": max_div["layer"],
                        "final_layer_divergence_pct": round((1.0 - final["cka"]) * 100.0, 3),
                        "final_layer": max_layer,
                        "late_mean_divergence_pct": round(float(np.mean([(1.0 - row["cka"]) * 100.0 for row in late])), 3),
                        "late_layers": f"{late_start}-{max_layer}",
                        "min_cka": round(max_div["cka"], 6),
                    }
                )
    return output


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "family",
        "source",
        "comparison",
        "max_divergence_pct",
        "max_divergence_layer",
        "final_layer_divergence_pct",
        "final_layer",
        "late_mean_divergence_pct",
        "late_layers",
        "min_cka",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def plot(rows: list[dict], out_base: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = sorted({row["family"] for row in rows})
    fig, axes = plt.subplots(len(families), 1, figsize=(7.0, 3.4 * len(families)), squeeze=False)
    for axis, family in zip(axes[:, 0], families):
        family_rows = [row for row in rows if row["family"] == family and row["source"] == "text"]
        grouped = group_rows(family_rows)
        for model_a, model_b, label in BASE_PAIRS:
            series = grouped.get(("text", model_a, model_b), [])
            if not series:
                continue
            axis.plot(
                [row["layer"] for row in series],
                [(1.0 - row["cka"]) * 100.0 for row in series],
                marker="o",
                linewidth=1.8,
                color=PAIR_COLORS[label],
                label=label,
            )
        axis.set_title(family)
        axis.set_xlabel("Layer")
        axis.set_ylabel("Divergence from base (%)")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lightweight paper artifacts from CKA JSON files.")
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--family", default="Qwen2-VL-7B")
    parser.add_argument("--output-dir", type=Path, default=Path("cka/results/paper"))
    parser.add_argument("--output-name", default="cka_image_conditioned_summary.csv")
    parser.add_argument("--write-figures", action="store_true", help="Also write a generated PNG divergence plot.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = load_rows(args.results, args.family)
    summary = summarize(all_rows)
    write_csv(summary, args.output_dir / args.output_name)
    if args.write_figures:
        plot(all_rows, args.output_dir / "cka_divergence_qwen")
    print(f"Wrote CKA artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
