"""
Phase 6 — Analytics primitives.

Three deliverables backed by AgentDecision logs:
  - objection_map:      Top REJECT/COMPLAIN themes (keyword clusters).
  - feature_importance: Feature mentions in BUY vs REJECT reasoning.
  - message_resonance:  Marketing-claim → fraction of BUY reasoning that echoes it.

Heavy NLP (BERTopic, sentence-transformers) is optional and lazy-loaded — the
defaults use lightweight token overlap so feasibility tests don't need ML deps.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

from launchlens.phase3.schemas import AgentDecision


_STOPWORDS = {
    "the", "and", "but", "for", "with", "this", "that", "from", "have", "not",
    "you", "your", "are", "was", "were", "will", "would", "should", "could",
    "can", "their", "they", "them", "what", "when", "which", "who", "why",
    "into", "out", "about", "than", "too", "very", "more", "less", "much",
    "given", "since", "because", "while", "still", "just", "also", "even",
    "i've", "i'm", "it's", "don't", "doesn't", "isn't", "wouldn't",
}


def _tokens(text: str) -> list[str]:
    out = []
    buf = []
    for ch in text.lower():
        if ch.isalnum() or ch == "'":
            buf.append(ch)
        else:
            if buf:
                t = "".join(buf)
                if len(t) > 3 and t not in _STOPWORDS:
                    out.append(t)
                buf = []
    if buf:
        t = "".join(buf)
        if len(t) > 3 and t not in _STOPWORDS:
            out.append(t)
    return out


# ── Objection map ────────────────────────────────────────────────────────────

def objection_map(
    decisions: Sequence[AgentDecision],
    top_n: int = 10,
) -> list[dict]:
    """
    Cluster REJECT/COMPLAIN reasoning by keyword frequency.
    Returns a list of {"keyword", "count", "example_reasons"} dicts.
    """
    rejects = [
        d for d in decisions
        if d.decision in ("REJECT", "COMPLAIN") and d.primary_reason.strip()
    ]
    if not rejects:
        return []

    keyword_counts: Counter[str] = Counter()
    keyword_examples: dict[str, list[str]] = defaultdict(list)

    for d in rejects:
        for tok in set(_tokens(d.primary_reason + " " + d.internal_reasoning)):
            keyword_counts[tok] += 1
            if len(keyword_examples[tok]) < 3:
                keyword_examples[tok].append(d.primary_reason[:120])

    return [
        {"keyword": kw, "count": count, "example_reasons": keyword_examples[kw]}
        for kw, count in keyword_counts.most_common(top_n)
    ]


# ── Feature importance ───────────────────────────────────────────────────────

def feature_importance(
    decisions: Sequence[AgentDecision],
    features: Sequence[str],
) -> list[dict]:
    """
    For each product feature, compute:
      - mentions_in_buy:    times the feature appears in BUY reasoning
      - mentions_in_reject: times the feature appears in REJECT/COMPLAIN reasoning
      - importance_score:   normalized (buy - reject) / total mentions, in [-1, 1]
    """
    buy_text = " ".join(
        (d.internal_reasoning + " " + d.primary_reason).lower()
        for d in decisions if d.decision == "BUY"
    )
    rej_text = " ".join(
        (d.internal_reasoning + " " + d.primary_reason).lower()
        for d in decisions if d.decision in ("REJECT", "COMPLAIN")
    )

    out = []
    for feature in features:
        feature_tokens = [t for t in _tokens(feature) if len(t) > 3]
        if not feature_tokens:
            continue
        buy_hits = sum(buy_text.count(t) for t in feature_tokens)
        rej_hits = sum(rej_text.count(t) for t in feature_tokens)
        total = buy_hits + rej_hits
        score = (buy_hits - rej_hits) / total if total else 0.0
        out.append({
            "feature": feature,
            "mentions_in_buy": buy_hits,
            "mentions_in_reject": rej_hits,
            "importance_score": round(score, 3),
        })
    out.sort(key=lambda x: -x["importance_score"])
    return out


# ── Message resonance ────────────────────────────────────────────────────────

def message_resonance(
    decisions: Sequence[AgentDecision],
    marketing_copy: str,
    threshold: int = 2,
) -> dict[str, float]:
    """
    For each substantive keyword in marketing_copy, return the fraction of
    BUY reasoning that contains it. Keywords with < `threshold` tokens are skipped.
    """
    copy_tokens = _tokens(marketing_copy)
    if not copy_tokens:
        return {}

    buy_reasoning = [
        (d.internal_reasoning + " " + d.primary_reason).lower()
        for d in decisions if d.decision == "BUY"
    ]
    if not buy_reasoning:
        return {t: 0.0 for t in copy_tokens}

    # Use distinct tokens, sorted by their order in the copy
    seen: list[str] = []
    for t in copy_tokens:
        if t not in seen:
            seen.append(t)

    out: dict[str, float] = {}
    for t in seen:
        hits = sum(1 for r in buy_reasoning if t in r)
        out[t] = round(hits / len(buy_reasoning), 3)
    return out


# ── Segment depth (lightweight K-means alternative) ──────────────────────────

def segment_breakdown(
    decisions: Sequence[AgentDecision],
    persona_segments: dict[str, str],
) -> list[dict]:
    """
    Per-segment decision distribution.
    `persona_segments` is a {agent_id: segment_label} map; build via your
    preferred clustering or the simple labels from phase5.calibration._segment_label.
    """
    latest: dict[str, str] = {}
    for d in sorted(decisions, key=lambda x: x.timestep):
        latest[d.agent_id] = d.decision

    by_seg: dict[str, Counter] = defaultdict(Counter)
    by_seg_total: Counter[str] = Counter()
    for aid, state in latest.items():
        seg = persona_segments.get(aid)
        if seg is None:
            continue
        by_seg[seg][state] += 1
        by_seg_total[seg] += 1

    out = []
    for seg, hist in by_seg.items():
        total = by_seg_total[seg]
        out.append({
            "segment": seg,
            "size": total,
            "buy_rate": round(hist.get("BUY", 0) / total, 3) if total else 0.0,
            "reject_rate": round(hist.get("REJECT", 0) / total, 3) if total else 0.0,
            "distribution": dict(hist),
        })
    out.sort(key=lambda x: -x["buy_rate"])
    return out
