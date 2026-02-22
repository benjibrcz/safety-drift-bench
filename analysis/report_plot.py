"""Generate a polished two-panel figure for reports/papers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from builders.trajectory_builder import EpisodeResult, TurnRecord
from metrics.hazard import kaplan_meier


def load_episodes(jsonl_path: Path) -> list[EpisodeResult]:
    episodes = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            turns = [TurnRecord(**t) for t in d["turns"]]
            episodes.append(
                EpisodeResult(
                    episode_id=d["episode_id"],
                    turn_count=d["turn_count"],
                    seed=d["seed"],
                    turns=turns,
                )
            )
    return episodes


def main(run_dir: str = "runs/quick_run"):
    run_path = PROJECT_ROOT / run_dir
    summary = pd.read_csv(run_path / "metrics" / "summary.csv")
    episodes = load_episodes(run_path / "logs" / "episodes.jsonl")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel A: Failure rate vs conversation length ──
    x = summary["turn_count"].values
    y = summary["failure_rate"].values
    lo = summary["failure_rate_ci_lo"].values
    hi = summary["failure_rate_ci_hi"].values

    ax1.errorbar(
        x, y,
        yerr=[y - lo, hi - y],
        fmt="o-",
        capsize=6,
        linewidth=2.2,
        markersize=9,
        color="#2563eb",
        markerfacecolor="#2563eb",
        markeredgecolor="white",
        markeredgewidth=1.5,
        ecolor="#93c5fd",
        elinewidth=1.8,
    )

    ax1.fill_between(x, lo, hi, alpha=0.12, color="#2563eb")
    ax1.set_xlabel("Conversation length (turns)", fontsize=12)
    ax1.set_ylabel("Episode failure rate", fontsize=12)
    ax1.set_title("A.  Safety-rule violation rate", fontsize=13, fontweight="bold", loc="left")
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_xticks(x)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelsize=10)

    # Annotate failure rates
    for xi, yi in zip(x, y):
        ax1.annotate(
            f"{yi:.0%}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            fontsize=11,
            fontweight="bold",
            color="#1e40af",
        )

    # ── Panel B: Kaplan–Meier survival curve ──
    km = kaplan_meier(episodes)

    ax2.step(
        km["turn"], km["survival"],
        where="post",
        linewidth=2.2,
        color="#dc2626",
    )
    ax2.fill_between(
        km["turn"], km["survival"], step="post",
        alpha=0.10, color="#dc2626",
    )

    ax2.set_xlabel("Turn index", fontsize=12)
    ax2.set_ylabel("P(no violation by turn t)", fontsize=12)
    ax2.set_title("B.  Time-to-first-failure survival", fontsize=13, fontweight="bold", loc="left")
    ax2.set_ylim(-0.05, 1.1)
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelsize=10)

    # Mark median survival
    below_half = km[km["survival"] <= 0.5]
    if len(below_half) > 0:
        median_turn = below_half.iloc[0]["turn"]
        ax2.axvline(x=median_turn, color="#dc2626", linestyle="--", alpha=0.5)
        ax2.annotate(
            f"median ≈ turn {int(median_turn)}",
            (median_turn, 0.5),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=10,
            color="#991b1b",
        )

    # ── Global ──
    fig.suptitle(
        "SafetyDriftBench: Llama 3.1 8B Instruct — \"never say apple\" rule",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5, -0.04,
        "5 episodes per bucket · p_probe = 0.3 · strict substring scoring · Wilson 95% CI",
        ha="center",
        fontsize=9,
        color="gray",
    )

    fig.tight_layout()
    out = run_path / "plots" / "report_figure.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved report figure → {out}")


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/quick_run"
    main(run_dir)
