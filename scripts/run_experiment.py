#!/usr/bin/env python3
"""Run the DriftBench experiment end-to-end.

Usage:
    python scripts/run_experiment.py --config configs/apple_rule.yaml
    python scripts/run_experiment.py --config configs/apple_rule.yaml --run-id my_run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.plot_results import plot_failure_rate, plot_survival_curve
from builders.trajectory_builder import TrajectoryBuilder
from metrics.summary import save_summary
from models.client import make_client, set_seeds


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run DriftBench experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/apple_rule.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (default: timestamp)",
    )
    args = parser.parse_args()

    # --- Load config ---
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp = config["experiment"]
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / config["logging"]["run_dir"] / run_id

    log_dir = run_dir / "logs"
    metrics_dir = run_dir / "metrics"
    plots_dir = run_dir / "plots"
    for d in [log_dir, metrics_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # --- Set seeds ---
    seed = exp.get("seed", 42)
    set_seeds(seed)

    # --- Build client ---
    print(f"Initializing model client (backend={config['model']['backend']})...")
    client = make_client(config)

    # --- Build trajectory runner ---
    builder = TrajectoryBuilder(
        client=client,
        system_prompt=config["system_prompt"],
        benign_path=PROJECT_ROOT / "data" / "prompts" / "benign_questions.txt",
        probe_path=PROJECT_ROOT / "data" / "prompts" / "probe_questions.txt",
        p_probe=exp.get("p_probe", 0.2),
        forbidden_word=config["scoring"]["forbidden_word"],
        strict_substring=config["scoring"]["strict_substring"],
    )

    # --- Run experiment ---
    turn_counts = exp["turn_counts"]
    episodes_per = exp["episodes_per_turn_count"]

    print(f"\nExperiment: {len(turn_counts)} turn-count buckets × {episodes_per} episodes each")
    print(f"Turn counts: {turn_counts}")
    print(f"Probe probability: {exp.get('p_probe', 0.2)}")
    print(f"Output: {run_dir}\n")

    episodes = builder.run_experiment(
        turn_counts=turn_counts,
        episodes_per_turn_count=episodes_per,
        base_seed=seed,
        log_dir=log_dir,
    )

    # --- Compute and save metrics ---
    print("\nComputing metrics...")
    summary_df, ep_df = save_summary(episodes, metrics_dir)

    # --- Generate plots ---
    print("Generating plots...")
    plot_failure_rate(summary_df, plots_dir / "failure_rate.png")
    plot_survival_curve(episodes, plots_dir / "survival_curve.png")

    # --- Print summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)
    print(f"\nAll outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
