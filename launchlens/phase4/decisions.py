"""Decision parser — extracts structured AgentDecision from raw LLM output.

Two-stage strategy:
1. JSON parse — the canonical format the prompt requests.
2. Fielded-text regex fallback — for models that disregard the JSON instruction.

If both fail, returns ``None`` and logs a structured ``decision_parse_failed`` event.
The caller (``phase4/loop.py``) increments ``TimestepLog.parse_failures`` rather than
coercing the response to ``IGNORE``. Silent fuzzy matching on the raw output has been
removed: it produced invisible BUYs when models hallucinated state tokens.
"""
from __future__ import annotations

import json
import re

import structlog

from launchlens.phase3.schemas import AgentDecision, DecisionState, DiscussTarget

log = structlog.get_logger()

_VALID_DECISIONS: set[str] = {
    "IGNORE", "AWARE", "RESEARCH", "CONSIDER",
    "BUY", "REJECT", "SHARE_POSITIVE", "SHARE_NEGATIVE", "COMPLAIN",
}
_VALID_DISCUSS: set[str] = {"family", "friends", "colleagues", "no_one"}

_REQUIRED_JSON_FIELDS = (
    "internal_reasoning",
    "decision",
    "primary_reason",
    "would_discuss_with",
    "language_of_discussion",
)

# Permissive JSON locator: first {...} block in the response.
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

# Legacy fielded-text format kept as fallback for older models.
_FIELD_RE = re.compile(
    r"INTERNAL_REASONING:\s*(.+?)\s*\n"
    r"DECISION:\s*(\w+)\s*\n"
    r"PRIMARY_REASON:\s*(.+?)\s*\n"
    r"WOULD_DISCUSS_WITH:\s*(\w+)\s*\n"
    r"LANGUAGE_OF_DISCUSSION:\s*(.+)",
    re.DOTALL,
)


class ParseFailure(Exception):
    """Raised internally when neither parse strategy succeeds."""


def _coerce_decision(value: str) -> DecisionState:
    v = value.strip().upper()
    if v not in _VALID_DECISIONS:
        raise ParseFailure(f"invalid decision token: {value!r}")
    return v  # type: ignore[return-value]


def _coerce_discuss(value: str) -> DiscussTarget:
    v = value.strip().lower()
    if v not in _VALID_DISCUSS:
        raise ParseFailure(f"invalid discuss token: {value!r}")
    return v  # type: ignore[return-value]


def _try_parse_json(raw: str) -> dict[str, str] | None:
    """Best-effort JSON parse of the first object in ``raw``."""
    match = _JSON_OBJ_RE.search(raw)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if not all(k in obj for k in _REQUIRED_JSON_FIELDS):
        return None
    return {k: str(obj[k]) for k in _REQUIRED_JSON_FIELDS}


def _try_parse_fielded(raw: str) -> dict[str, str] | None:
    """Fallback: parse the legacy KEY: value newline-delimited format."""
    match = _FIELD_RE.search(raw)
    if not match:
        return None
    reasoning, decision_raw, reason, discuss_raw, lang = match.groups()
    return {
        "internal_reasoning": reasoning.strip(),
        "decision": decision_raw.strip(),
        "primary_reason": reason.strip(),
        "would_discuss_with": discuss_raw.strip(),
        "language_of_discussion": lang.strip(),
    }


def parse_decision(
    raw: str,
    agent_id: str,
    product_id: str,
    timestep: int,
) -> AgentDecision | None:
    """Parse a raw LLM response into an ``AgentDecision``.

    Returns ``None`` if the response cannot be unambiguously parsed; callers
    should treat ``None`` as a failure and record it (do not silently coerce
    to ``IGNORE``).
    """
    fields = _try_parse_json(raw) or _try_parse_fielded(raw)
    if fields is None:
        log.warning(
            "decision_parse_failed",
            agent_id=agent_id,
            product_id=product_id,
            timestep=timestep,
            sample=raw[:160],
        )
        return None

    try:
        decision = _coerce_decision(fields["decision"])
        discuss = _coerce_discuss(fields["would_discuss_with"])
    except ParseFailure as exc:
        log.warning(
            "decision_value_invalid",
            agent_id=agent_id,
            product_id=product_id,
            timestep=timestep,
            error=str(exc),
            sample=raw[:160],
        )
        return None

    return AgentDecision(
        agent_id=agent_id,
        product_id=product_id,
        timestep=timestep,
        internal_reasoning=fields["internal_reasoning"].strip(),
        decision=decision,
        primary_reason=fields["primary_reason"].strip(),
        would_discuss_with=discuss,
        language_of_discussion=fields["language_of_discussion"].strip() or "N/A",
    )
