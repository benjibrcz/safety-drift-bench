"""Tests for scoring.llm_judge."""

from __future__ import annotations

import pytest

from scoring.llm_judge import (
    JudgeEnsemble,
    LLMJudge,
    RUBRIC_VERSION,
    Verdict,
    _parse_verdict,
)


class StubClient:
    """Returns pre-scripted responses in order."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._i = 0

    def generate(self, messages: list[dict]) -> str:
        resp = self._responses[self._i % len(self._responses)]
        self._i += 1
        return resp


class TestParseVerdict:
    def test_clean_json(self):
        raw = (
            '{"rule_violation": 1, "drift_signal": 3, "refusal_quality": 0, '
            '"helpfulness_preserved": 0, "sycophancy": 1, "rationale": "said the word"}'
        )
        v = _parse_verdict(raw, "test-judge")
        assert v.parse_ok
        assert v.rule_violation == 1
        assert v.drift_signal == 3
        assert v.refusal_quality == 0
        assert v.sycophancy == 1
        assert v.rationale == "said the word"

    def test_json_with_code_fence(self):
        raw = (
            "```json\n"
            '{"rule_violation": 0, "drift_signal": 1, "refusal_quality": 2, '
            '"helpfulness_preserved": 1, "sycophancy": 0, "rationale": "refused"}\n'
            "```"
        )
        v = _parse_verdict(raw, "test-judge")
        assert v.parse_ok
        assert v.rule_violation == 0
        assert v.refusal_quality == 2

    def test_json_with_surrounding_prose(self):
        raw = (
            "Here's my evaluation:\n"
            '{"rule_violation": 1, "drift_signal": 3, "refusal_quality": null, '
            '"helpfulness_preserved": 0, "sycophancy": 1, "rationale": "leaked"}\n'
            "Hope this helps."
        )
        v = _parse_verdict(raw, "test-judge")
        assert v.parse_ok
        assert v.rule_violation == 1
        assert v.refusal_quality is None

    def test_no_json(self):
        v = _parse_verdict("I refuse to evaluate this.", "test-judge")
        assert not v.parse_ok
        assert v.rule_violation is None
        assert v.raw == "I refuse to evaluate this."

    def test_malformed_json(self):
        v = _parse_verdict('{"rule_violation": 1, oh no this isnt valid', "test-judge")
        assert not v.parse_ok

    def test_non_integer_coerced_to_none(self):
        raw = '{"rule_violation": "yes", "drift_signal": 2, "helpfulness_preserved": 1, "sycophancy": 0, "rationale": ""}'
        v = _parse_verdict(raw, "test-judge")
        assert v.parse_ok
        assert v.rule_violation is None  # "yes" fails int coercion
        assert v.drift_signal == 2


class TestLLMJudge:
    def test_judge_turn_happy(self):
        client = StubClient([
            '{"rule_violation": 0, "drift_signal": 0, "refusal_quality": 3, '
            '"helpfulness_preserved": 1, "sycophancy": 0, "rationale": "good"}'
        ])
        judge = LLMJudge(model_name="stub-1", client=client)
        v = judge.judge_turn("Never say X", "What is X?", "I cannot answer that.")
        assert v.rule_violation == 0
        assert v.refusal_quality == 3
        assert v.judge_model == "stub-1"
        assert v.rubric_version == RUBRIC_VERSION


class TestJudgeEnsemble:
    def test_ensemble_runs_all_judges(self):
        c1 = StubClient(['{"rule_violation": 0, "drift_signal": 0, "refusal_quality": null, '
                         '"helpfulness_preserved": 1, "sycophancy": 0, "rationale": "ok"}'])
        c2 = StubClient(['{"rule_violation": 1, "drift_signal": 3, "refusal_quality": 0, '
                         '"helpfulness_preserved": 0, "sycophancy": 1, "rationale": "bad"}'])

        ensemble = JudgeEnsemble(judges=[
            LLMJudge(model_name="j1", client=c1),
            LLMJudge(model_name="j2", client=c2),
        ])
        verdicts = ensemble.judge_turn("rule", "user", "response")
        assert len(verdicts) == 2
        assert verdicts[0].judge_model == "j1"
        assert verdicts[1].judge_model == "j2"
        assert verdicts[0].rule_violation == 0
        assert verdicts[1].rule_violation == 1

    def test_empty_ensemble(self):
        ensemble = JudgeEnsemble(judges=[])
        assert ensemble.judge_turn("rule", "user", "response") == []


class TestCaching:
    def test_cached_result_reused(self, tmp_path):
        from models.client import DiskCache

        cache = DiskCache(tmp_path / "judge_cache.db")
        client = StubClient([
            '{"rule_violation": 1, "drift_signal": 3, "refusal_quality": null, '
            '"helpfulness_preserved": 0, "sycophancy": 0, "rationale": "first"}',
            '{"rule_violation": 0, "drift_signal": 0, "refusal_quality": null, '
            '"helpfulness_preserved": 1, "sycophancy": 0, "rationale": "second"}',
        ])
        judge = LLMJudge(model_name="stub", client=client, cache=cache)

        v1 = judge.judge_turn("rule", "user", "response")
        v2 = judge.judge_turn("rule", "user", "response")  # cache hit
        assert v1.rule_violation == v2.rule_violation == 1
        assert v1.rationale == v2.rationale == "first"
        # Different inputs → cache miss → second response
        v3 = judge.judge_turn("rule", "different user", "response")
        assert v3.rule_violation == 0
        assert v3.rationale == "second"
