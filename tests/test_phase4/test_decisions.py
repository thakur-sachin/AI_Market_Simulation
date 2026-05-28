"""Decision parser tests — JSON-first with fielded-text fallback.

We intentionally do NOT silently coerce invalid decisions to IGNORE; tests below
pin that contract. Loop callers count returned ``None`` as parse_failure.
"""
from launchlens.phase4.decisions import parse_decision


def test_strict_fielded_format_parses():
    raw = (
        "INTERNAL_REASONING: I have considered the price and find it acceptable.\n"
        "DECISION: BUY\n"
        "PRIMARY_REASON: Good value for money.\n"
        "WOULD_DISCUSS_WITH: friends\n"
        "LANGUAGE_OF_DISCUSSION: hindi"
    )
    d = parse_decision(raw, "a1", "p1", 0)
    assert d is not None
    assert d.decision == "BUY"
    assert d.would_discuss_with == "friends"
    assert d.language_of_discussion == "hindi"


def test_json_format_parses():
    raw = (
        '{"internal_reasoning":"price is right",'
        '"decision":"BUY",'
        '"primary_reason":"good value",'
        '"would_discuss_with":"friends",'
        '"language_of_discussion":"hindi"}'
    )
    d = parse_decision(raw, "a1", "p1", 0)
    assert d is not None
    assert d.decision == "BUY"


def test_json_with_surrounding_prose():
    """LLMs sometimes wrap the JSON in prose. The locator regex finds the object."""
    raw = (
        "Sure, here's my response:\n"
        '{"internal_reasoning":"r","decision":"REJECT",'
        '"primary_reason":"too pricey","would_discuss_with":"family",'
        '"language_of_discussion":"hindi"}\n'
        "Hope that helps!"
    )
    d = parse_decision(raw, "a1", "p1", 0)
    assert d is not None
    assert d.decision == "REJECT"


def test_compound_token_share_positive():
    """SHARE_POSITIVE is a valid token; bare 'POSITIVE' is not."""
    raw = (
        "INTERNAL_REASONING: I love it.\n"
        "DECISION: SHARE_POSITIVE\n"
        "PRIMARY_REASON: Telling friends.\n"
        "WOULD_DISCUSS_WITH: friends\n"
        "LANGUAGE_OF_DISCUSSION: english"
    )
    d = parse_decision(raw, "a1", "p1", 0)
    assert d is not None
    assert d.decision == "SHARE_POSITIVE"


def test_malformed_returns_none():
    raw = "I don't know what to do."
    d = parse_decision(raw, "a1", "p1", 0)
    assert d is None


def test_invalid_decision_token_returns_none():
    """No silent IGNORE coercion. Invalid → None (caller counts as parse_failure)."""
    raw = (
        '{"internal_reasoning":"r","decision":"maybe",'
        '"primary_reason":"hmm","would_discuss_with":"friends",'
        '"language_of_discussion":"english"}'
    )
    d = parse_decision(raw, "a1", "p1", 0)
    assert d is None


def test_invalid_discuss_target_returns_none():
    """Strict on discuss target too."""
    raw = (
        '{"internal_reasoning":"r","decision":"IGNORE",'
        '"primary_reason":"r","would_discuss_with":"nobody_in_particular",'
        '"language_of_discussion":"english"}'
    )
    d = parse_decision(raw, "a1", "p1", 0)
    assert d is None
