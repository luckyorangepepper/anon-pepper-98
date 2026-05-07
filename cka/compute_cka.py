from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml


PAIRS = [
    ("base", "text-sft"),
    ("base", "vision-sft"),
    ("base", "full-sft"),
    ("text-sft", "vision-sft"),
    ("text-sft", "full-sft"),
]


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    numerator = np.linalg.norm(y.T @ x, "fro") ** 2
    denominator = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    if denominator < 1e-10:
        return 0.0
    return float(numerator / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute layerwise linear CKA from saved hidden-state tensors.")
    parser.add_argument("--hidden-states-dir", type=Path, required=True)
    parser.add_argument("--layers-config", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--source", default="text")
    args = parser.parse_args()

    layers = yaml.safe_load(args.layers_config.read_text(encoding="utf-8"))["layers"]
    results = []
    for model_a, model_b in PAIRS:
        for layer in layers:
            path_a = args.hidden_states_dir / f"{model_a}_layer{layer}.pt"
            path_b = args.hidden_states_dir / f"{model_b}_layer{layer}.pt"
            if not path_a.exists() or not path_b.exists():
                continue
            x = torch.load(path_a, weights_only=True).numpy()
            y = torch.load(path_b, weights_only=True).numpy()
            results.append(
                {
                    "source": args.source,
                    "model_a": model_a,
                    "model_b": model_b,
                    "layer": layer,
                    "cka": round(linear_cka(x, y), 6),
                }
            )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} CKA rows to {args.output_file}")


if __name__ == "__main__":
    main()
