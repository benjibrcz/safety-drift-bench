"""Tests for builders.adversaries."""

from __future__ import annotations

import random

import pytest

from builders.adversaries import STRATEGIES, get_adversary


class TestStrategies:
    def test_all_strategies_have_nonempty_pools(self):
        for name, strat in STRATEGIES.items():
            assert len(strat.templates) > 0, f"{name} has empty template pool"
            assert all(isinstance(t, str) and t for t in strat.templates)

    def test_strategy_name_matches_key(self):
        for name, strat in STRATEGIES.items():
            assert strat.name == name

    def test_get_adversary_known(self):
        adv = get_adversary("direct")
        assert adv.name == "direct"

    def test_get_adversary_unknown(self):
        with pytest.raises(ValueError, match="Unknown elicitation strategy"):
            get_adversary("nonexistent")


class TestDeterminism:
    """Same RNG seed → same probe sequence (required for factorial reproducibility)."""

    def test_same_seed_same_sequence(self):
        adv = get_adversary("authority")
        rng_a = random.Random(123)
        rng_b = random.Random(123)
        seq_a = [adv.probe(rng_a) for _ in range(10)]
        seq_b = [adv.probe(rng_b) for _ in range(10)]
        assert seq_a == seq_b

    def test_different_seed_different_sequence(self):
        adv = get_adversary("roleplay")
        rng_a = random.Random(1)
        rng_b = random.Random(2)
        seq_a = [adv.probe(rng_a) for _ in range(20)]
        seq_b = [adv.probe(rng_b) for _ in range(20)]
        assert seq_a != seq_b

    def test_probes_drawn_from_strategy_pool(self):
        """Every drawn probe must come from the strategy's declared pool."""
        rng = random.Random(0)
        for name, strat in STRATEGIES.items():
            for _ in range(5):
                probe = strat.probe(rng)
                assert probe in strat.templates, f"{name}: {probe!r} not in pool"


class TestStrategyDisjointness:
    """Basic sanity: strategies shouldn't accidentally share templates — that
    would mean varying the 'strategy' axis doesn't actually vary anything."""

    def test_direct_and_authority_disjoint(self):
        direct_set = set(STRATEGIES["direct"].templates)
        auth_set = set(STRATEGIES["authority"].templates)
        assert direct_set.isdisjoint(auth_set)

    def test_direct_and_hypothetical_disjoint(self):
        d = set(STRATEGIES["direct"].templates)
        h = set(STRATEGIES["hypothetical"].templates)
        assert d.isdisjoint(h)
