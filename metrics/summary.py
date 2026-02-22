"""Aggregate metrics from episode results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from builders.trajectory_builder import EpisodeResult


def episodes_to_dataframe(episodes: list[EpisodeResult]) -> pd.DataFrame:
    """Convert a list of EpisodeResult into a tidy DataFrame (one row per episode)."""
    rows = []
    for ep in episodes:
        rows.append(
            {
                "episode_id": ep.episode_id,
                "turn_count": ep.turn_count,
                "seed": ep.seed,
                "failed": ep.failed,
                "time_to_first_failure": ep.time_to_first_failure,
                "n_violations": sum(1 for t in ep.turns if t.violated_rule),
                "n_probes": sum(1 for t in ep.turns if t.is_probe),
                "n_probe_violations": sum(
                    1 for t in ep.turns if t.is_probe and t.violated_rule
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-turn-count summary statistics.

    Returns a DataFrame with columns:
        turn_count, n_episodes, failure_rate, failure_rate_ci_lo, failure_rate_ci_hi,
        mean_ttff, median_ttff, probe_violation_rate
    """
    records = []
    for T, grp in df.groupby("turn_count"):
        n = len(grp)
        k = grp["failed"].sum()
        p = k / n if n else 0.0

        # Wilson score interval for proportion
        lo, hi = _wilson_ci(k, n)

        # Time-to-first-failure stats (conditional on failure)
        failed = grp.loc[grp["failed"], "time_to_first_failure"]
        mean_ttff = failed.mean() if len(failed) else None
        median_ttff = failed.median() if len(failed) else None

        # Probe violation rate
        total_probes = grp["n_probes"].sum()
        total_probe_violations = grp["n_probe_violations"].sum()
        probe_viol_rate = (
            total_probe_violations / total_probes if total_probes else 0.0
        )

        records.append(
            {
                "turn_count": T,
                "n_episodes": n,
                "failure_rate": p,
                "failure_rate_ci_lo": lo,
                "failure_rate_ci_hi": hi,
                "mean_ttff": mean_ttff,
                "median_ttff": median_ttff,
                "probe_violation_rate": probe_viol_rate,
            }
        )

    return pd.DataFrame(records)


def save_summary(
    episodes: list[EpisodeResult],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute and write summary.csv + episodes.csv to *output_dir*.

    Returns (summary_df, episodes_df).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ep_df = episodes_to_dataframe(episodes)
    summary_df = compute_summary(ep_df)

    ep_df.to_csv(output_dir / "episodes.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    return summary_df, ep_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))
