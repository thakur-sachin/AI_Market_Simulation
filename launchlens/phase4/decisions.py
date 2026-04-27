"""Decision parser — extracts structured AgentDecision from raw LLM output."""
from __future__ import annotations

import re

import structlog

from launchlens.phase3.schemas import AgentDecision, DecisionState, DiscussTarget

log = structlog.get_logger()

_VALID_DECISIONS: set[str] = {
    "IGNORE", "AWARE", "RESEARCH", "CONSIDER",
    "BUY", "REJECT", "SHARE_POSITIVE", "SHARE_NEGATIVE", "COMPLAIN",
}
_VALID_DISCUSS: set[str] = {"family", "friends", "colleagues", "no_one"}

_FIELD_RE = re.compile(
    r"INTERNAL_REASONING:\s*(.+?)\s*\n"
    r"DECISION:\s*(\w+)\s*\n"
    r"PRIMARY_REASON:\s*(.+?)\s*\n"
    r"WOULD_DISCUSS_WITH:\s*(\w+)\s*\n"
    r"LANGUAGE_OF_DISCUSSION:\s*(.+)",
    re.DOTALL,
)


def parse_decision(
    raw: str,
    agent_id: str,
    product_id: str,
    timestep: int,
) -> AgentDecision | None:
    match = _FIELD_RE.search(raw)
    if not match:
        log.warning("decision_parse_failed", agent_id=agent_id, raw=raw[:120])
        return None

    reasoning, decision_raw, reason, discuss_raw, lang = match.groups()
    decision_raw = decision_raw.strip().upper()
    discuss_raw = discuss_raw.strip().lower()

    if decision_raw not in _VALID_DECISIONS:
        # Fuzzy fallback: find nearest valid token in raw output
        for token in _VALID_DECISIONS:
            if token in raw.upper():
                decision_raw = token
                break
        else:
            decision_raw = "IGNORE"

    if discuss_raw not in _VALID_DISCUSS:
        discuss_raw = "no_one"

    return AgentDecision(
        agent_id=agent_id,
        product_id=product_id,
        timestep=timestep,
        internal_reasoning=reasoning.strip(),
        decision=decision_raw,   # type: ignore[arg-type]
        primary_reason=reason.strip(),
        would_discuss_with=discuss_raw,  # type: ignore[arg-type]
        language_of_discussion=lang.strip(),
    )
