"""Inter-rater agreement metrics for the LLM judge ensemble.

A single LLM judge is noisy. Running an ensemble and reporting agreement
tells you how *reliable* any one dimension's score is — if two strong
judges barely agree, that dimension is likely underspecified (or the
task genuinely ambiguous), and single-judge numbers should not be
trusted as-is.

Functions
---------
cohens_kappa     — two raters, binary or ordinal labels
fleiss_kappa     — N ≥ 2 raters, categorical labels, equal raters per item
percent_agreement — raw % of items where all raters agree
agreement_table  — full table across rubric dimensions for a list of
                   verdict-lists (one per turn)
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pairwise and multi-rater kappa
# ---------------------------------------------------------------------------

def cohens_kappa(
    rater_a: Sequence[Optional[int]],
    rater_b: Sequence[Optional[int]],
) -> float:
    """Cohen's kappa for two raters on categorical labels.

    Items where either rater returned ``None`` (unscorable / parse failure)
    are dropped before the computation.  Returns ``float('nan')`` if fewer
    than two items remain or if there is no variance.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must be the same length")

    pairs = [(a, b) for a, b in zip(rater_a, rater_b) if a is not None and b is not None]
    if len(pairs) < 2:
        return float("nan")

    a_arr = np.array([p[0] for p in pairs])
    b_arr = np.array([p[1] for p in pairs])

    categories = sorted(set(a_arr.tolist()) | set(b_arr.tolist()))
    n = len(pairs)

    observed = float((a_arr == b_arr).mean())
    expected = 0.0
    for c in categories:
        p_a = float((a_arr == c).mean())
        p_b = float((b_arr == c).mean())
        expected += p_a * p_b

    if expected >= 1.0:
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def fleiss_kappa(ratings: Sequence[Sequence[Optional[int]]]) -> float:
    """Fleiss' kappa for N raters on categorical labels.

    ``ratings`` is a matrix of shape (n_items, n_raters).  Items where ANY
    rater is ``None`` are dropped (Fleiss requires equal raters per item).
    """
    clean = [list(row) for row in ratings if all(v is not None for v in row)]
    if len(clean) < 2:
        return float("nan")

    n_raters = len(clean[0])
    if any(len(row) != n_raters for row in clean):
        raise ValueError("All items must have the same number of raters")

    categories = sorted({v for row in clean for v in row})
    cat_idx = {c: i for i, c in enumerate(categories)}

    n_items = len(clean)
    counts = np.zeros((n_items, len(categories)), dtype=int)
    for i, row in enumerate(clean):
        for v in row:
            counts[i, cat_idx[v]] += 1

    p_j = counts.sum(axis=0) / (n_items * n_raters)  # marginal category freq
    if n_raters == 1:
        return float("nan")
    P_i = (np.square(counts).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    Pe = float(np.square(p_j).sum())

    if Pe >= 1.0:
        return float("nan")
    return (P_bar - Pe) / (1.0 - Pe)


def percent_agreement(ratings: Sequence[Sequence[Optional[int]]]) -> float:
    """Fraction of items where all raters produced the same non-null label."""
    clean = [row for row in ratings if all(v is not None for v in row)]
    if not clean:
        return float("nan")
    return sum(len(set(row)) == 1 for row in clean) / len(clean)


# ---------------------------------------------------------------------------
# Rubric-wide table
# ---------------------------------------------------------------------------

_DIMENSIONS = (
    "rule_violation",
    "drift_signal",
    "refusal_quality",
    "helpfulness_preserved",
    "sycophancy",
)


def agreement_table(
    verdicts_per_turn: Sequence[Sequence["Verdict"]],  # noqa: F821 — fwd ref
) -> pd.DataFrame:
    """Compute per-dimension agreement across an ensemble.

    ``verdicts_per_turn[i]`` is the list of verdicts from all judges on
    turn *i* (same judge order in every row).  Returns a DataFrame with
    one row per dimension and columns:

        dimension, n_items, percent_agreement, fleiss_kappa, mean_pairwise_kappa
    """
    if not verdicts_per_turn:
        return pd.DataFrame(columns=[
            "dimension", "n_items", "percent_agreement",
            "fleiss_kappa", "mean_pairwise_kappa",
        ])

    n_judges = len(verdicts_per_turn[0])
    rows = []
    for dim in _DIMENSIONS:
        matrix = [
            [getattr(v, dim) for v in turn_verdicts]
            for turn_verdicts in verdicts_per_turn
        ]
        n_items = sum(1 for row in matrix if all(v is not None for v in row))

        if n_judges >= 2:
            pairwise: list[float] = []
            for i in range(n_judges):
                for j in range(i + 1, n_judges):
                    k = cohens_kappa(
                        [row[i] for row in matrix],
                        [row[j] for row in matrix],
                    )
                    if not np.isnan(k):
                        pairwise.append(k)
            mean_pairwise = float(np.mean(pairwise)) if pairwise else float("nan")
        else:
            mean_pairwise = float("nan")

        rows.append({
            "dimension": dim,
            "n_items": n_items,
            "percent_agreement": percent_agreement(matrix),
            "fleiss_kappa": fleiss_kappa(matrix) if n_judges >= 2 else float("nan"),
            "mean_pairwise_kappa": mean_pairwise,
        })

    return pd.DataFrame(rows)


def judge_vs_deterministic(
    verdicts_per_turn: Sequence[Sequence["Verdict"]],  # noqa: F821
    deterministic_labels: Sequence[int],
) -> pd.DataFrame:
    """Kappa between each judge's ``rule_violation`` score and the
    deterministic substring scorer — a sanity floor for the binary
    dimension. A judge that doesn't beat the substring scorer is adding
    only noise on that dimension.
    """
    if not verdicts_per_turn:
        return pd.DataFrame(columns=["judge_model", "cohens_kappa_vs_hard"])

    n_judges = len(verdicts_per_turn[0])
    rows = []
    for j in range(n_judges):
        judge_model = verdicts_per_turn[0][j].judge_model
        labels = [turn[j].rule_violation for turn in verdicts_per_turn]
        k = cohens_kappa(labels, list(deterministic_labels))
        rows.append({"judge_model": judge_model, "cohens_kappa_vs_hard": k})
    return pd.DataFrame(rows)
