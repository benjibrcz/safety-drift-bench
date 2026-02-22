"""Survival / hazard analysis for time-to-first-failure."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from builders.trajectory_builder import EpisodeResult


def kaplan_meier(
    episodes: list[EpisodeResult],
    max_turn: Optional[int] = None,
) -> pd.DataFrame:
    """Compute a Kaplan–Meier survival table from episode results.

    Each episode is treated as a subject.  The "event" is first rule violation;
    episodes that never fail are right-censored at their total turn count.

    Returns a DataFrame with columns: turn, n_at_risk, n_events, survival.
    """
    # Build event times
    times: list[int] = []
    censored: list[bool] = []
    for ep in episodes:
        ttff = ep.time_to_first_failure
        if ttff is not None:
            times.append(ttff)
            censored.append(False)
        else:
            times.append(ep.turn_count)
            censored.append(True)

    if max_turn is None:
        max_turn = max(times) if times else 0

    # Sort by time
    order = np.argsort(times)
    times_arr = np.array(times)[order]
    censored_arr = np.array(censored)[order]

    n = len(times_arr)
    at_risk = n
    survival = 1.0

    rows = [{"turn": 0, "n_at_risk": at_risk, "n_events": 0, "survival": survival}]

    i = 0
    while i < n:
        t = times_arr[i]
        # Count events and censorings at this time
        n_events = 0
        n_censored = 0
        while i < n and times_arr[i] == t:
            if censored_arr[i]:
                n_censored += 1
            else:
                n_events += 1
            i += 1

        if n_events > 0 and at_risk > 0:
            survival *= 1.0 - n_events / at_risk

        rows.append(
            {
                "turn": int(t),
                "n_at_risk": at_risk,
                "n_events": n_events,
                "survival": survival,
            }
        )

        at_risk -= n_events + n_censored

    return pd.DataFrame(rows)
