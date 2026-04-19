#!/usr/bin/env python3
"""Run a factorial SafetyDriftBench experiment.

Each cell in the grid = (elicitation_strategy × turn_count).  Holding every
other axis fixed (model, sampling, p_probe, seed offset) means any observed
difference in failure rate is attributable to the axis being varied — not
to a noisy, free-form adversary.

Outputs
-------
runs/<run_id>/
    config.yaml
    cells/
        strategy=direct/
            logs/episodes.jsonl
            metrics/{summary.csv,episodes.csv}
    grid_summary.csv     # one row per (strategy, turn_count)

Usage
-----
    python scripts/run_factorial.py --config configs/factorial.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from builders.adversaries import get_adversary
from builders.trajectory_builder import TrajectoryBuilder
from metrics.summary import compute_summary, episodes_to_dataframe
from models.client import make_client, set_seeds


def _run_cell(
    builder: TrajectoryBuilder,
    strategy_name: str,
    turn_counts: list[int],
    episodes_per_cell: int,
    base_seed: int,
    cell_dir: Path,
) -> pd.DataFrame:
    """Run all episodes for one strategy, across all turn_counts, and
    return its per-turn-count summary DataFrame (with strategy column)."""
    log_dir = cell_dir / "logs"
    metrics_dir = cell_dir / "metrics"
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Seed offset per strategy so cells are independent but reproducible.
    strategy_offset = abs(hash(strategy_name)) % 1_000_000
    episodes = builder.run_experiment(
        turn_counts=turn_counts,
        episodes_per_turn_count=episodes_per_cell,
        base_seed=base_seed + strategy_offset,
        log_dir=log_dir,
    )

    ep_df = episodes_to_dataframe(episodes)
    summary_df = compute_summary(ep_df)
    ep_df.to_csv(metrics_dir / "episodes.csv", index=False)
    summary_df.to_csv(metrics_dir / "summary.csv", index=False)

    summary_df = summary_df.copy()
    summary_df.insert(0, "strategy", strategy_name)
    return summary_df


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run a factorial DriftBench experiment")
    parser.add_argument("--config", type=str, default="configs/factorial.yaml")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "factorial" not in config:
        raise ValueError(
            f"Config {config_path} has no `factorial:` section. "
            f"Use scripts/run_experiment.py for single-cell configs."
        )

    fac = config["factorial"]
    strategies: list[str] = fac["strategies"]
    turn_counts: list[int] = fac["turn_counts"]
    episodes_per_cell: int = fac["episodes_per_cell"]
    p_probe: float = fac.get("p_probe", 0.2)
    seed: int = fac.get("seed", 42)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / config["logging"]["run_dir"] / run_id
    cells_dir = run_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    set_seeds(seed)

    print(f"Factorial grid: {len(strategies)} strategies × {len(turn_counts)} turn_counts")
    print(f"  strategies:   {strategies}")
    print(f"  turn_counts:  {turn_counts}")
    print(f"  episodes/cell: {episodes_per_cell}  (total episodes = "
          f"{len(strategies) * len(turn_counts) * episodes_per_cell})")
    print(f"  output:       {run_dir}\n")

    print(f"Initializing model client (backend={config['model']['backend']})...")
    client = make_client(config)

    all_summaries: list[pd.DataFrame] = []
    for strategy_name in strategies:
        adversary = get_adversary(strategy_name)
        builder = TrajectoryBuilder(
            client=client,
            system_prompt=config["system_prompt"],
            benign_path=PROJECT_ROOT / "data" / "prompts" / "benign_questions.txt",
            probe_path=PROJECT_ROOT / "data" / "prompts" / "probe_questions.txt",
            p_probe=p_probe,
            forbidden_word=config["scoring"]["forbidden_word"],
            strict_substring=config["scoring"]["strict_substring"],
            adversary=adversary,
        )

        cell_dir = cells_dir / f"strategy={strategy_name}"
        print(f"\n--- Cell: strategy={strategy_name} ---")
        summary_df = _run_cell(
            builder=builder,
            strategy_name=strategy_name,
            turn_counts=turn_counts,
            episodes_per_cell=episodes_per_cell,
            base_seed=seed,
            cell_dir=cell_dir,
        )
        all_summaries.append(summary_df)

    grid = pd.concat(all_summaries, ignore_index=True)
    grid.to_csv(run_dir / "grid_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("GRID SUMMARY")
    print("=" * 70)
    print(grid.to_string(index=False))
    print("=" * 70)
    print(f"\nAll outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
