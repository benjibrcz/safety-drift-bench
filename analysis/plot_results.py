"""Plotting utilities for DriftBench results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from builders.trajectory_builder import EpisodeResult
from metrics.hazard import kaplan_meier


def plot_failure_rate(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot failure rate vs conversation length with Wilson CIs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = summary_df["turn_count"]
    y = summary_df["failure_rate"]
    lo = summary_df["failure_rate_ci_lo"]
    hi = summary_df["failure_rate_ci_hi"]

    ax.errorbar(
        x,
        y,
        yerr=[y - lo, hi - y],
        fmt="o-",
        capsize=5,
        linewidth=2,
        markersize=8,
        color="#2563eb",
    )

    ax.set_xlabel("Conversation length (turns)", fontsize=13)
    ax.set_ylabel("Episode failure rate", fontsize=13)
    ax.set_title("Safety-rule failure rate vs. conversation length", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xscale("log")
    ax.set_xticks(sorted(x))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved failure-rate plot → {output_path}")


def plot_survival_curve(
    episodes: list[EpisodeResult],
    output_path: str | Path,
) -> None:
    """Plot Kaplan–Meier survival curve (probability of *not* failing by turn t)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    km = kaplan_meier(episodes)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(km["turn"], km["survival"], where="post", linewidth=2, color="#dc2626")

    ax.set_xlabel("Turn index", fontsize=13)
    ax.set_ylabel("P(no failure by turn t)", fontsize=13)
    ax.set_title("Kaplan–Meier survival curve (time to first failure)", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved survival-curve plot → {output_path}")
