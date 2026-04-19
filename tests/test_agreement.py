"""Tests for metrics.agreement."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metrics.agreement import (
    agreement_table,
    cohens_kappa,
    fleiss_kappa,
    judge_vs_deterministic,
    percent_agreement,
)
from scoring.llm_judge import RUBRIC_VERSION, Verdict


def _v(model: str, **scores) -> Verdict:
    return Verdict(judge_model=model, rubric_version=RUBRIC_VERSION, **scores)


class TestCohensKappa:
    def test_perfect_agreement(self):
        a = [0, 1, 0, 1, 1, 0]
        b = [0, 1, 0, 1, 1, 0]
        assert cohens_kappa(a, b) == pytest.approx(1.0)

    def test_perfect_disagreement(self):
        a = [0, 1, 0, 1]
        b = [1, 0, 1, 0]
        # p_o=0, p_e depends on marginals; with balanced labels p_e=0.5 → k=-1
        assert cohens_kappa(a, b) == pytest.approx(-1.0)

    def test_chance_agreement_zero(self):
        # Both raters label 0 half the time, 1 half the time, independently.
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, size=2000).tolist()
        b = rng.integers(0, 2, size=2000).tolist()
        k = cohens_kappa(a, b)
        assert abs(k) < 0.1  # near zero

    def test_known_value(self):
        # n=100, p_o=0.8, balanced marginals (p_e=0.5) → kappa=0.6
        a = [1] * 50 + [0] * 50
        b = [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40
        assert cohens_kappa(a, b) == pytest.approx(0.6, abs=1e-9)

    def test_drops_nulls(self):
        a = [0, None, 1, 1]
        b = [0, 1, 1, None]
        # After dropping: a=[0,1], b=[0,1] → perfect
        assert cohens_kappa(a, b) == pytest.approx(1.0)

    def test_too_few_items(self):
        assert math.isnan(cohens_kappa([1], [1]))
        assert math.isnan(cohens_kappa([None, 1], [1, None]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cohens_kappa([0, 1], [0, 1, 0])


class TestFleissKappa:
    def test_perfect_agreement(self):
        # 3 raters all agree on every item
        ratings = [[1, 1, 1], [0, 0, 0], [1, 1, 1]]
        assert fleiss_kappa(ratings) == pytest.approx(1.0)

    def test_no_agreement(self):
        # Max possible disagreement within a row
        ratings = [[0, 1, 2], [0, 1, 2], [2, 0, 1]]
        k = fleiss_kappa(ratings)
        assert k < 0 or k == pytest.approx(0.0, abs=0.3)

    def test_drops_rows_with_null(self):
        ratings = [
            [1, 1, 1],
            [None, 1, 1],   # dropped
            [0, 0, 0],
        ]
        assert fleiss_kappa(ratings) == pytest.approx(1.0)

    def test_uneven_raters_raises(self):
        with pytest.raises(ValueError):
            fleiss_kappa([[1, 1], [0, 0, 0]])


class TestPercentAgreement:
    def test_all_agree(self):
        assert percent_agreement([[1, 1, 1], [0, 0, 0]]) == pytest.approx(1.0)

    def test_half_agree(self):
        assert percent_agreement([[1, 1, 1], [0, 1, 0]]) == pytest.approx(0.5)

    def test_empty(self):
        assert math.isnan(percent_agreement([]))


class TestAgreementTable:
    def test_shape_and_columns(self):
        # Two judges, three turns, all agreeing perfectly
        vpt = [
            [_v("A", rule_violation=0, drift_signal=0, refusal_quality=None,
                helpfulness_preserved=1, sycophancy=0),
             _v("B", rule_violation=0, drift_signal=0, refusal_quality=None,
                helpfulness_preserved=1, sycophancy=0)],
            [_v("A", rule_violation=1, drift_signal=3, refusal_quality=0,
                helpfulness_preserved=0, sycophancy=1),
             _v("B", rule_violation=1, drift_signal=3, refusal_quality=0,
                helpfulness_preserved=0, sycophancy=1)],
        ]
        df = agreement_table(vpt)
        assert set(df["dimension"]) == {
            "rule_violation", "drift_signal", "refusal_quality",
            "helpfulness_preserved", "sycophancy",
        }
        # rule_violation is binary + perfect agreement → percent=1, kappa=1
        row = df[df["dimension"] == "rule_violation"].iloc[0]
        assert row["percent_agreement"] == pytest.approx(1.0)
        assert row["fleiss_kappa"] == pytest.approx(1.0)

    def test_disagreement_lowers_kappa(self):
        # Judges disagree on rule_violation half the time → kappa ~ 0
        vpt = [
            [_v("A", rule_violation=0), _v("B", rule_violation=1)],
            [_v("A", rule_violation=1), _v("B", rule_violation=0)],
            [_v("A", rule_violation=0), _v("B", rule_violation=0)],
            [_v("A", rule_violation=1), _v("B", rule_violation=1)],
        ]
        df = agreement_table(vpt)
        row = df[df["dimension"] == "rule_violation"].iloc[0]
        assert row["percent_agreement"] == pytest.approx(0.5)
        assert -0.5 < row["fleiss_kappa"] < 0.5


class TestJudgeVsDeterministic:
    def test_perfect_alignment(self):
        vpt = [
            [_v("A", rule_violation=0)],
            [_v("A", rule_violation=1)],
            [_v("A", rule_violation=1)],
        ]
        hard = [0, 1, 1]
        df = judge_vs_deterministic(vpt, hard)
        assert df.iloc[0]["cohens_kappa_vs_hard"] == pytest.approx(1.0)

    def test_complete_disagreement(self):
        vpt = [
            [_v("A", rule_violation=0)],
            [_v("A", rule_violation=1)],
        ]
        hard = [1, 0]
        df = judge_vs_deterministic(vpt, hard)
        assert df.iloc[0]["cohens_kappa_vs_hard"] == pytest.approx(-1.0)
