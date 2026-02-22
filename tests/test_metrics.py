"""Tests for metrics.summary and metrics.hazard."""

import pytest

from builders.trajectory_builder import EpisodeResult, TurnRecord
from metrics.hazard import kaplan_meier
import pandas as pd

from metrics.summary import _wilson_ci, compute_summary, episodes_to_dataframe


def _make_episode(
    episode_id: str,
    turn_count: int,
    failure_turn: int | None = None,
) -> EpisodeResult:
    """Helper: build a synthetic episode with optional failure at a given turn."""
    turns = []
    for t in range(turn_count):
        violated = failure_turn is not None and t == failure_turn
        turns.append(
            TurnRecord(
                turn_idx=t,
                is_probe=t % 3 == 0,  # every 3rd turn is a probe
                user_text=f"question {t}",
                assistant_text="apple" if violated else "safe",
                violated_rule=violated,
            )
        )
    return EpisodeResult(
        episode_id=episode_id,
        turn_count=turn_count,
        seed=0,
        turns=turns,
    )


class TestEpisodesToDataframe:
    def test_basic(self):
        episodes = [
            _make_episode("ep0", 10, failure_turn=5),
            _make_episode("ep1", 10, failure_turn=None),
        ]
        df = episodes_to_dataframe(episodes)
        assert len(df) == 2
        assert bool(df.loc[0, "failed"]) is True
        assert df.loc[0, "time_to_first_failure"] == 5
        assert bool(df.loc[1, "failed"]) is False
        assert pd.isna(df.loc[1, "time_to_first_failure"])


class TestComputeSummary:
    def test_all_fail(self):
        episodes = [_make_episode(f"ep{i}", 10, failure_turn=i) for i in range(5)]
        df = episodes_to_dataframe(episodes)
        summary = compute_summary(df)
        assert len(summary) == 1
        assert summary.iloc[0]["failure_rate"] == 1.0

    def test_none_fail(self):
        episodes = [_make_episode(f"ep{i}", 10) for i in range(5)]
        df = episodes_to_dataframe(episodes)
        summary = compute_summary(df)
        assert summary.iloc[0]["failure_rate"] == 0.0

    def test_mixed(self):
        episodes = [
            _make_episode("ep0", 10, failure_turn=3),
            _make_episode("ep1", 10),
            _make_episode("ep2", 10, failure_turn=7),
            _make_episode("ep3", 10),
        ]
        df = episodes_to_dataframe(episodes)
        summary = compute_summary(df)
        assert summary.iloc[0]["failure_rate"] == 0.5

    def test_multiple_turn_counts(self):
        episodes = [
            _make_episode("a0", 10, failure_turn=2),
            _make_episode("a1", 10),
            _make_episode("b0", 25, failure_turn=10),
            _make_episode("b1", 25, failure_turn=20),
        ]
        df = episodes_to_dataframe(episodes)
        summary = compute_summary(df)
        assert len(summary) == 2
        row10 = summary[summary["turn_count"] == 10].iloc[0]
        row25 = summary[summary["turn_count"] == 25].iloc[0]
        assert row10["failure_rate"] == 0.5
        assert row25["failure_rate"] == 1.0


class TestWilsonCI:
    def test_zero_of_zero(self):
        lo, hi = _wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 0.0

    def test_all_success(self):
        lo, hi = _wilson_ci(0, 100)
        assert lo == 0.0
        assert hi < 0.05

    def test_all_fail(self):
        lo, hi = _wilson_ci(100, 100)
        assert lo > 0.95
        assert hi >= 0.999

    def test_half(self):
        lo, hi = _wilson_ci(50, 100)
        assert 0.35 < lo < 0.5
        assert 0.5 < hi < 0.65


class TestKaplanMeier:
    def test_no_failures(self):
        episodes = [_make_episode(f"ep{i}", 10) for i in range(3)]
        km = kaplan_meier(episodes)
        # Survival should stay at 1.0 throughout
        assert km.iloc[-1]["survival"] == 1.0

    def test_all_fail_at_same_time(self):
        episodes = [_make_episode(f"ep{i}", 10, failure_turn=5) for i in range(3)]
        km = kaplan_meier(episodes)
        # After turn 5, survival should be 0
        last = km[km["turn"] >= 5].iloc[0]
        assert last["survival"] == 0.0

    def test_staggered_failures(self):
        episodes = [
            _make_episode("ep0", 10, failure_turn=2),
            _make_episode("ep1", 10, failure_turn=5),
            _make_episode("ep2", 10),  # never fails (censored)
        ]
        km = kaplan_meier(episodes)
        # Survival should decrease at turns 2 and 5
        assert km.iloc[0]["survival"] == 1.0  # turn 0
        # After some events, survival < 1
        final = km.iloc[-1]["survival"]
        assert 0 < final < 1.0
