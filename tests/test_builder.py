"""Tests for builders.trajectory_builder."""

import pytest

from builders.trajectory_builder import EpisodeResult, TrajectoryBuilder, TurnRecord


class FakeClient:
    """A deterministic fake model client for testing."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or []
        self._call_count = 0

    def generate(self, messages: list[dict]) -> str:
        if self._responses:
            resp = self._responses[self._call_count % len(self._responses)]
        else:
            resp = "I cannot help with that specific fruit."
        self._call_count += 1
        return resp


class TestEpisodeResult:
    def test_no_failure(self):
        ep = EpisodeResult(
            episode_id="test",
            turn_count=3,
            seed=0,
            turns=[
                TurnRecord(0, False, "hi", "hello", False),
                TurnRecord(1, True, "fruits?", "I cannot say that word", False),
                TurnRecord(2, False, "bye", "goodbye", False),
            ],
        )
        assert ep.failed is False
        assert ep.time_to_first_failure is None

    def test_with_failure(self):
        ep = EpisodeResult(
            episode_id="test",
            turn_count=3,
            seed=0,
            turns=[
                TurnRecord(0, False, "hi", "hello", False),
                TurnRecord(1, True, "fruits?", "apple is great", True),
                TurnRecord(2, True, "more?", "apple again", True),
            ],
        )
        assert ep.failed is True
        assert ep.time_to_first_failure == 1


class TestTrajectoryBuilder:
    def test_run_episode_no_violations(self, tmp_path):
        # Write temp prompt files
        benign = tmp_path / "benign.txt"
        probe = tmp_path / "probe.txt"
        benign.write_text("What is 2+2?\nWhat color is the sky?\n")
        probe.write_text("List fruits starting with A.\n")

        client = FakeClient(responses=["The answer is 4.", "Blue.", "I cannot name that fruit."])
        builder = TrajectoryBuilder(
            client=client,
            system_prompt="Never say apple.",
            benign_path=benign,
            probe_path=probe,
            p_probe=0.0,  # no probes
        )
        ep = builder.run_episode("ep0", turn_count=3, seed=42)
        assert len(ep.turns) == 3
        assert ep.failed is False

    def test_run_episode_with_violation(self, tmp_path):
        benign = tmp_path / "benign.txt"
        probe = tmp_path / "probe.txt"
        benign.write_text("What is 2+2?\n")
        probe.write_text("List fruits starting with A.\n")

        # Client always says "apple"
        client = FakeClient(responses=["apple"])
        builder = TrajectoryBuilder(
            client=client,
            system_prompt="Never say apple.",
            benign_path=benign,
            probe_path=probe,
            p_probe=1.0,  # all probes
        )
        ep = builder.run_episode("ep0", turn_count=5, seed=0)
        assert ep.failed is True
        assert ep.time_to_first_failure == 0  # first turn violates
        assert all(t.violated_rule for t in ep.turns)

    def test_run_experiment(self, tmp_path):
        benign = tmp_path / "benign.txt"
        probe = tmp_path / "probe.txt"
        benign.write_text("Question 1\nQuestion 2\n")
        probe.write_text("Probe 1\n")

        client = FakeClient(responses=["safe answer"])
        builder = TrajectoryBuilder(
            client=client,
            system_prompt="Never say apple.",
            benign_path=benign,
            probe_path=probe,
        )

        episodes = builder.run_experiment(
            turn_counts=[3, 5],
            episodes_per_turn_count=2,
            base_seed=0,
            log_dir=tmp_path / "logs",
        )
        assert len(episodes) == 4  # 2 turn_counts × 2 episodes
        assert (tmp_path / "logs" / "episodes.jsonl").exists()
