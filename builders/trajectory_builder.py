"""Trajectory builder — runs multi-turn conversations and records results."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from models.client import ModelClient, set_seeds
from scoring.hard_constraint import contains_forbidden_word


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    turn_idx: int
    is_probe: bool
    user_text: str
    assistant_text: str
    violated_rule: bool


@dataclass
class EpisodeResult:
    episode_id: str
    turn_count: int
    seed: int
    turns: list[TurnRecord] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(t.violated_rule for t in self.turns)

    @property
    def time_to_first_failure(self) -> Optional[int]:
        for t in self.turns:
            if t.violated_rule:
                return t.turn_idx
        return None


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_lines(path: str | Path) -> list[str]:
    """Load non-empty lines from a text file."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class TrajectoryBuilder:
    """Orchestrates multi-turn conversations for one experiment config."""

    def __init__(
        self,
        client: ModelClient,
        system_prompt: str,
        benign_path: str | Path,
        probe_path: str | Path,
        p_probe: float = 0.2,
        forbidden_word: str = "apple",
        strict_substring: bool = True,
    ):
        self.client = client
        self.system_prompt = system_prompt
        self.benign_questions = _load_lines(benign_path)
        self.probe_questions = _load_lines(probe_path)
        self.p_probe = p_probe
        self.forbidden_word = forbidden_word
        self.strict_substring = strict_substring

    def run_episode(
        self,
        episode_id: str,
        turn_count: int,
        seed: int,
    ) -> EpisodeResult:
        """Run a single multi-turn conversation and return structured results."""
        set_seeds(seed)

        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        episode = EpisodeResult(episode_id=episode_id, turn_count=turn_count, seed=seed)

        for t in range(turn_count):
            is_probe = random.random() < self.p_probe
            if is_probe:
                user_text = random.choice(self.probe_questions)
            else:
                user_text = random.choice(self.benign_questions)

            messages.append({"role": "user", "content": user_text})
            assistant_text = self.client.generate(messages)
            messages.append({"role": "assistant", "content": assistant_text})

            violated = contains_forbidden_word(
                assistant_text,
                word=self.forbidden_word,
                strict_substring=self.strict_substring,
            )

            episode.turns.append(
                TurnRecord(
                    turn_idx=t,
                    is_probe=is_probe,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    violated_rule=violated,
                )
            )

        return episode

    def run_experiment(
        self,
        turn_counts: list[int],
        episodes_per_turn_count: int,
        base_seed: int = 42,
        log_dir: Optional[str | Path] = None,
    ) -> list[EpisodeResult]:
        """Run the full experiment across all turn-count buckets."""
        all_episodes: list[EpisodeResult] = []

        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "episodes.jsonl"
            log_handle = open(log_file, "w")
        else:
            log_handle = None

        total = sum(episodes_per_turn_count for _ in turn_counts)
        pbar = tqdm(total=total, desc="Running episodes")

        for T in turn_counts:
            for ep_idx in range(episodes_per_turn_count):
                seed = base_seed + T * 10000 + ep_idx
                episode_id = f"T{T}_ep{ep_idx}"
                episode = self.run_episode(episode_id, T, seed)
                all_episodes.append(episode)

                if log_handle:
                    log_handle.write(json.dumps(asdict(episode)) + "\n")
                    log_handle.flush()

                pbar.update(1)

        pbar.close()
        if log_handle:
            log_handle.close()

        return all_episodes
