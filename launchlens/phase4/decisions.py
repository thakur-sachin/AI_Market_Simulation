"""Decision parser — extracts structured AgentDecision from raw LLM output.

Strategy:
  1. Try strict regex (the format the prompt asks for).
  2. Fall back to line-by-line key extraction (tolerates markdown, extra whitespace,
     reordered fields, headers like ``**DECISION**``).
  3. Last-ditch: scan for any valid decision token, prefer the one that follows
     the literal "DECISION" if multiple appear.
  4. If even the decision token is unrecoverable, return None — caller treats as
     a parse failure (distinct from LLM errors).
"""
from __future__ import annotations

import re

import structlog

from launchlens.phase3.schemas import AgentDecision

log = structlog.get_logger()

_VALID_DECISIONS: tuple[str, ...] = (
    "SHARE_POSITIVE", "SHARE_NEGATIVE",   # check compound tokens first
    "IGNORE", "AWARE", "RESEARCH", "CONSIDER",
    "BUY", "REJECT", "COMPLAIN",
)
_VALID_DISCUSS: tuple[str, ...] = ("family", "friends", "colleagues", "no_one")

_STRICT_RE = re.compile(
    r"INTERNAL_REASONING:\s*(.+?)\s*\n"
    r"DECISION:\s*(\w+)\s*\n"
    r"PRIMARY_REASON:\s*(.+?)\s*\n"
    r"WOULD_DISCUSS_WITH:\s*(\w+)\s*\n"
    r"LANGUAGE_OF_DISCUSSION:\s*(.+)",
    re.DOTALL,
)

# Lenient field extractor: "** DECISION ** : BUY"  or  "Decision - BUY"
def _extract_field(raw: str, key: str) -> str | None:
    pattern = re.compile(
        rf"(?:\*\*\s*)?{key}(?:\s*\*\*)?\s*[:\-]\s*(.+?)(?:\n|$)",
        re.IGNORECASE,
    )
    m = pattern.search(raw)
    return m.group(1).strip().strip("*").strip() if m else None


def _coerce_decision(text: str | None, raw: str) -> str | None:
    """Try to find a valid decision token. Prefers the value of the DECISION field."""
    candidates: list[str] = []
    if text:
        candidates.append(text.upper())
    candidates.append(raw.upper())

    for src in candidates:
        for token in _VALID_DECISIONS:
            # word-boundary match; tolerates surrounding punctuation
            if re.search(rf"\b{token}\b", src):
                return token
    return None


def _coerce_discuss(text: str | None) -> str:
    if not text:
        return "no_one"
    t = text.lower().strip()
    for token in _VALID_DISCUSS:
        if token in t:
            return token
    return "no_one"


def parse_decision(
    raw: str,
    agent_id: str,
    product_id: str,
    timestep: int,
) -> AgentDecision | None:
    raw = raw.strip()

    # Strict path
    m = _STRICT_RE.search(raw)
    if m:
        reasoning, decision_raw, reason, discuss_raw, lang = (g.strip() for g in m.groups())
        decision = _coerce_decision(decision_raw, raw)
        if decision is None:
            log.warning("decision_token_invalid", agent_id=agent_id, raw=raw[:120])
            return None
        return AgentDecision(
            agent_id=agent_id,
            product_id=product_id,
            timestep=timestep,
            internal_reasoning=reasoning,
            decision=decision,                       # type: ignore[arg-type]
            primary_reason=reason,
            would_discuss_with=_coerce_discuss(discuss_raw),  # type: ignore[arg-type]
            language_of_discussion=lang,
        )

    # Lenient path
    reasoning = _extract_field(raw, "INTERNAL_REASONING") or _extract_field(raw, "reasoning") or ""
    decision_field = _extract_field(raw, "DECISION") or _extract_field(raw, "state")
    reason = _extract_field(raw, "PRIMARY_REASON") or _extract_field(raw, "reason") or ""
    discuss = _extract_field(raw, "WOULD_DISCUSS_WITH") or _extract_field(raw, "discuss")
    lang = _extract_field(raw, "LANGUAGE_OF_DISCUSSION") or _extract_field(raw, "language") or "N/A"

    decision = _coerce_decision(decision_field, raw)
    if decision is None:
        log.warning("decision_parse_failed", agent_id=agent_id, raw=raw[:120])
        return None

    # If reasoning wasn't extracted, use the first ~200 chars as best-effort context
    if not reasoning:
        reasoning = raw[:200]

    return AgentDecision(
        agent_id=agent_id,
        product_id=product_id,
        timestep=timestep,
        internal_reasoning=reasoning.strip(),
        decision=decision,                            # type: ignore[arg-type]
        primary_reason=reason.strip() or "(no reason given)",
        would_discuss_with=_coerce_discuss(discuss),  # type: ignore[arg-type]
        language_of_discussion=lang.strip(),
    )
