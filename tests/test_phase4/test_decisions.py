"""Decision parser tests — exercise both the JSON and fielded-text paths."""
from __future__ import annotations

import json

import pytest

from launchlens.phase4.decisions import parse_decision


def _wf_json(**overrides) -> str:
    payload = {
        "internal_reasoning": "Inner monologue about price and need.",
        "decision": "RESEARCH",
        "primary_reason": "Want to read reviews first.",
        "would_discuss_with": "friends",
        "language_of_discussion": "Hindi",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_parse_json_well_formed():
    out = parse_decision(_wf_json(), "agent_001", "prod_x", 0)
    assert out is not None
    assert out.decision == "RESEARCH"
    assert out.would_discuss_with == "friends"
    assert out.language_of_discussion == "Hindi"


def test_parse_json_with_surrounding_prose():
    raw = "Sure, here is the response:\n" + _wf_json(decision="BUY") + "\nThanks."
    out = parse_decision(raw, "a", "p", 0)
    assert out is not None
    assert out.decision == "BUY"


def test_parse_json_markdown_fenced():
    raw = "```json\n" + _wf_json(decision="REJECT") + "\n```"
    out = parse_decision(raw, "a", "p", 0)
    assert out is not None
    assert out.decision == "REJECT"


def test_parse_json_lowercase_decision_coerced():
    raw = _wf_json(decision="buy")
    out = parse_decision(raw, "a", "p", 0)
    assert out is not None
    assert out.decision == "BUY"


def test_parse_json_invalid_decision_returns_none():
    raw = _wf_json(decision="MAYBE")
    out = parse_decision(raw, "a", "p", 0)
    assert out is None


def test_parse_json_invalid_discuss_returns_none():
    raw = _wf_json(would_discuss_with="strangers")
    out = parse_decision(raw, "a", "p", 0)
    assert out is None


def test_parse_json_missing_field_returns_none():
    raw = json.dumps({
        "internal_reasoning": "x",
        "decision": "IGNORE",
        # primary_reason missing
        "would_discuss_with": "no_one",
        "language_of_discussion": "Hindi",
    })
    out = parse_decision(raw, "a", "p", 0)
    assert out is None


def test_parse_fielded_text_fallback():
    raw = (
        "INTERNAL_REASONING: I'm not sure yet, need more info.\n"
        "DECISION: AWARE\n"
        "PRIMARY_REASON: Heard about it briefly.\n"
        "WOULD_DISCUSS_WITH: family\n"
        "LANGUAGE_OF_DISCUSSION: Hindi"
    )
    out = parse_decision(raw, "a", "p", 0)
    assert out is not None
    assert out.decision == "AWARE"
    assert out.would_discuss_with == "family"


def test_parse_fielded_extra_whitespace():
    raw = (
        "INTERNAL_REASONING:   plenty of whitespace  \n"
        "DECISION:   CONSIDER  \n"
        "PRIMARY_REASON: ok\n"
        "WOULD_DISCUSS_WITH:  friends \n"
        "LANGUAGE_OF_DISCUSSION: English"
    )
    out = parse_decision(raw, "a", "p", 0)
    assert out is not None
    assert out.decision == "CONSIDER"


def test_parse_malformed_returns_none_not_ignore():
    """Critical: no silent fuzzy fallback. Failure must be observable."""
    raw = "I think the product is fine. Maybe buy. No JSON here, no fields."
    out = parse_decision(raw, "a", "p", 0)
    assert out is None


def test_parse_empty_string_returns_none():
    assert parse_decision("", "a", "p", 0) is None


def test_parse_only_partial_json_returns_none():
    raw = '{"decision": "BUY", "primary_reason":'  # truncated
    assert parse_decision(raw, "a", "p", 0) is None


def test_parse_preserves_timestep_and_ids():
    out = parse_decision(_wf_json(), "agent_42", "prod_xyz", 7)
    assert out is not None
    assert out.agent_id == "agent_42"
    assert out.product_id == "prod_xyz"
    assert out.timestep == 7
