"""Decision parser tests — covers strict, lenient, and fuzzy paths."""
from launchlens.phase4.decisions import parse_decision


def test_strict_format_parses():
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


def test_lenient_markdown_format():
    """LLM might wrap fields in markdown bold."""
    raw = (
        "**INTERNAL_REASONING:** I am skeptical of new health brands.\n"
        "**DECISION:** REJECT\n"
        "**PRIMARY_REASON:** Too expensive.\n"
        "**WOULD_DISCUSS_WITH:** family\n"
        "**LANGUAGE_OF_DISCUSSION:** hindi"
    )
    d = parse_decision(raw, "a1", "p1", 0)
    assert d is not None
    assert d.decision == "REJECT"


def test_fuzzy_decision_token_in_text():
    """When DECISION field has garbage, look for valid token in raw output."""
    raw = (
        "INTERNAL_REASONING: I will RESEARCH this further.\n"
        "DECISION: maybe\n"
        "PRIMARY_REASON: Need more info.\n"
        "WOULD_DISCUSS_WITH: friends\n"
        "LANGUAGE_OF_DISCUSSION: english"
    )
    d = parse_decision(raw, "a1", "p1", 0)
    assert d is not None
    assert d.decision == "RESEARCH"


def test_compound_token_share_positive():
    """SHARE_POSITIVE must be detected even though 'POSITIVE' alone isn't valid."""
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


def test_invalid_discuss_target_defaults_to_no_one():
    raw = (
        "INTERNAL_REASONING: Meh.\n"
        "DECISION: IGNORE\n"
        "PRIMARY_REASON: Don't care.\n"
        "WOULD_DISCUSS_WITH: nobody_in_particular\n"
        "LANGUAGE_OF_DISCUSSION: english"
    )
    d = parse_decision(raw, "a1", "p1", 0)
    assert d is not None
    assert d.would_discuss_with == "no_one"
