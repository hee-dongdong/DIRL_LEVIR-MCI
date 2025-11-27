#!/usr/bin/env python3
"""Parse training logs and plot loss curves saved as an image."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

LOSS_KEYS = ("speaker_loss", "cdcr_loss", "sim_loss", "aux_loss")

FLOAT_PATTERN = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_log_file(log_path: Path) -> List[Tuple[int, Dict[str, float]]]:
    """Return a sorted list of (epoch, losses) pairs parsed from the log."""
    epoch_pattern = re.compile(r"\[Epoch:\s*(\d+)")
    loss_patterns = {
        key: re.compile(rf"{key}:\s*({FLOAT_PATTERN})") for key in LOSS_KEYS
    }

    epoch_losses: Dict[int, Dict[str, float]] = {}

    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            epoch_match = epoch_pattern.search(line)
            if not epoch_match:
                continue
            epoch = int(epoch_match.group(1))
            line_losses = {}
            for key, pattern in loss_patterns.items():
                match = pattern.search(line)
                if match:
                    line_losses[key] = float(match.group(1))
            if not line_losses:
                continue
            # Keep the latest entry per epoch in case multiple matches exist.
            epoch_losses[epoch] = {**epoch_losses.get(epoch, {}), **line_losses}

    if not epoch_losses:
        raise ValueError(
            f"No losses found in {log_path}. "
            "Ensure the log contains '[Epoch:' lines with loss values."
        )

    return sorted(epoch_losses.items(), key=lambda item: item[0])


def plot_losses(
    epoch_loss_pairs: List[Tuple[int, Dict[str, float]]], output_path: Path
) -> None:
    """Create and save subplots for each available loss type."""
    epochs = [epoch for epoch, _ in epoch_loss_pairs]
    available_keys = [
        key
        for key in LOSS_KEYS
        if any(key in losses for _, losses in epoch_loss_pairs)
    ]

    if not available_keys:
        raise ValueError("No recognizable loss keys found to plot.")

    n_plots = len(available_keys)
    fig, axes = plt.subplots(
        n_plots, 1, figsize=(8, 3 * n_plots), sharex=True, constrained_layout=True
    )
    if n_plots == 1:
        axes = [axes]

    for ax, key in zip(axes, available_keys):
        values = [
            losses.get(key, math.nan) for _, losses in epoch_loss_pairs
        ]
        ax.plot(epochs, values, label=key, color="tab:blue")
        ax.set_ylabel("Loss")
        ax.set_title(key.replace("_", " ").title())
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Epoch")

    fig.suptitle("Loss Curves", y=1.02, fontsize=14)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a training log and plot per-epoch loss curves."
    )
    parser.add_argument(
        "log_path",
        type=Path,
        help="Path to the log file (e.g., logs/Out_folder/example.out).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the output PNG. Defaults to <log_name>_loss.png.",
    )
    args = parser.parse_args()

    log_path = args.log_path.expanduser().resolve()
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    output_path = (
        args.output
        if args.output is not None
        else log_path.with_suffix("").with_name(f"{log_path.stem}_loss.png")
    )

    epoch_loss_pairs = parse_log_file(log_path)
    plot_losses(epoch_loss_pairs, output_path)
    print(f"Saved loss curves to: {output_path}")


if __name__ == "__main__":
    main()
