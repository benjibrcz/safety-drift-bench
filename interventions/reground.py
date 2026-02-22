"""Stub for re-grounding interventions.

Future work: re-inject the system prompt at intervals to test whether
periodic reminders reduce drift.
"""

from __future__ import annotations


def reground_system_prompt(
    messages: list[dict],
    system_prompt: str,
    every_n_turns: int = 10,
    current_turn: int = 0,
) -> list[dict]:
    """Optionally re-inject the system prompt as a user-role reminder.

    This is a stub — not wired into the main experiment loop yet.
    """
    if current_turn > 0 and current_turn % every_n_turns == 0:
        reminder = (
            f"[System reminder] Please remember: {system_prompt}"
        )
        messages.append({"role": "user", "content": reminder})
    return messages
