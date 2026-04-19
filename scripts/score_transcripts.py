#!/usr/bin/env python3
"""Post-hoc transcript scoring via the LLM judge ensemble.

Runs independently of the main experiment loop so it can re-score any
existing ``runs/<run_id>/logs/episodes.jsonl`` without re-running the
subject model.  Outputs:

    runs/<run_id>/judgments.jsonl   — one row per (episode, turn, judge)
    runs/<run_id>/agreement.csv     — per-dimension agreement table
    runs/<run_id>/judge_vs_hard.csv — kappa vs deterministic scorer
    runs/<run_id>/judge_summary.csv — per-judge mean scores

Usage
-----
    python scripts/score_transcripts.py \\
        --run-dir runs/20260419_154000 \\
        --judge-config configs/judges.yaml \\
        --probes-only --episode-sample 20
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.agreement import (
    agreement_table,
    judge_vs_deterministic,
)
from scoring.llm_judge import Verdict, build_ensemble_from_config


def _iter_turns(
    episodes_path: Path,
    probes_only: bool,
    max_turns_per_episode: int | None,
    episode_sample: int | None,
    seed: int,
) -> Iterable[tuple[str, dict]]:
    """Yield ``(episode_id, turn_record)`` tuples from an episodes.jsonl file."""
    import random

    episodes = []
    with open(episodes_path) as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    if episode_sample is not None and episode_sample < len(episodes):
        rng = random.Random(seed)
        episodes = rng.sample(episodes, episode_sample)

    for ep in episodes:
        turns = ep["turns"]
        if probes_only:
            turns = [t for t in turns if t.get("is_probe")]
        if max_turns_per_episode is not None:
            turns = turns[:max_turns_per_episode]
        for t in turns:
            yield ep["episode_id"], t


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Post-hoc transcript scoring")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to runs/<run_id> directory")
    parser.add_argument("--judge-config", type=str, required=True,
                        help="YAML with a `judge:` section listing judges")
    parser.add_argument("--probes-only", action="store_true",
                        help="Only judge probe turns (default: all turns)")
    parser.add_argument("--max-turns-per-episode", type=int, default=None,
                        help="Cap turns per episode (default: all)")
    parser.add_argument("--episode-sample", type=int, default=None,
                        help="Randomly sample this many episodes (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    episodes_path = run_dir / "logs" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"No episodes.jsonl at {episodes_path}")

    # --- Load subject config to recover the safety rule ---
    subject_cfg_path = run_dir / "config.yaml"
    with open(subject_cfg_path) as f:
        subject_cfg = yaml.safe_load(f)
    safety_rule = subject_cfg["system_prompt"]

    # --- Build judge ensemble ---
    with open(args.judge_config) as f:
        judge_cfg = yaml.safe_load(f)
    cache_path = judge_cfg.get("judge", {}).get("cache_path")
    ensemble = build_ensemble_from_config(judge_cfg["judge"], cache_path=cache_path)
    if not ensemble.judges:
        raise ValueError("judge.judges is empty — configure at least one judge")
    print(f"Ensemble: {[j.model_name for j in ensemble.judges]}")

    # --- Collect turns to judge ---
    turns = list(_iter_turns(
        episodes_path,
        probes_only=args.probes_only,
        max_turns_per_episode=args.max_turns_per_episode,
        episode_sample=args.episode_sample,
        seed=args.seed,
    ))
    print(f"Judging {len(turns)} turn(s) with {len(ensemble.judges)} judge(s)...")

    # --- Run judges ---
    judgments_path = run_dir / "judgments.jsonl"
    verdicts_per_turn: list[list[Verdict]] = []
    hard_labels: list[int] = []

    with open(judgments_path, "w") as out:
        for episode_id, turn in tqdm(turns, desc="judging"):
            verdicts = ensemble.judge_turn(
                rule=safety_rule,
                user_text=turn["user_text"],
                assistant_text=turn["assistant_text"],
            )
            verdicts_per_turn.append(verdicts)
            hard_labels.append(int(bool(turn["violated_rule"])))
            for v in verdicts:
                out.write(json.dumps({
                    "episode_id": episode_id,
                    "turn_idx": turn["turn_idx"],
                    "is_probe": turn.get("is_probe"),
                    "hard_violation": int(bool(turn["violated_rule"])),
                    **asdict(v),
                }) + "\n")

    # --- Agreement + sanity-floor tables ---
    agreement_df = agreement_table(verdicts_per_turn)
    agreement_df.to_csv(run_dir / "agreement.csv", index=False)

    hard_df = judge_vs_deterministic(verdicts_per_turn, hard_labels)
    hard_df.to_csv(run_dir / "judge_vs_hard.csv", index=False)

    # --- Per-judge mean scores (for a quick at-a-glance comparison) ---
    summary_rows = []
    n_judges = len(ensemble.judges)
    for j in range(n_judges):
        judge_model = verdicts_per_turn[0][j].judge_model if verdicts_per_turn else \
            ensemble.judges[j].model_name
        per_dim = {}
        for dim in ("rule_violation", "drift_signal", "refusal_quality",
                    "helpfulness_preserved", "sycophancy"):
            values = [getattr(turn[j], dim) for turn in verdicts_per_turn]
            non_null = [v for v in values if v is not None]
            per_dim[f"mean_{dim}"] = (sum(non_null) / len(non_null)) if non_null else None
            per_dim[f"n_{dim}"] = len(non_null)
        summary_rows.append({"judge_model": judge_model, **per_dim})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "judge_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("JUDGE AGREEMENT")
    print("=" * 70)
    print(agreement_df.to_string(index=False))
    print("\nKappa vs deterministic substring scorer:")
    print(hard_df.to_string(index=False))
    print("\nPer-judge means:")
    print(summary_df.to_string(index=False))
    print(f"\nWrote: {judgments_path}")
    print(f"       {run_dir / 'agreement.csv'}")
    print(f"       {run_dir / 'judge_vs_hard.csv'}")
    print(f"       {run_dir / 'judge_summary.csv'}")


if __name__ == "__main__":
    main()
