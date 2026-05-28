"""Decision prompt builder — converts agent context + feed into LLM input.

Output contract: the model is asked to return a single JSON object whose fields
match ``_REQUIRED_JSON_FIELDS`` in ``decisions.py``. We also accept the legacy
fielded-text format as a fallback (handled by the parser).

Embeds an *anti-positivity prior* aligned with research on Indian consumer
price sensitivity (mitigates the LLM tendency to over-generate BUY).
"""
from __future__ import annotations

from launchlens.phase3.feed import render_feed_text
from launchlens.phase3.schemas import AgentMemory, MarketplaceFeed

_DECISION_SYSTEM = """\
You are role-playing as the specific Indian consumer described below. Stay fully in \
character at all times. Do not break character, do not add meta-commentary, do not \
explain that you are an AI.

Your responses must reflect this person's actual knowledge, biases, language patterns, \
and decision-making style. Use first-person reasoning rooted in their economic reality.

IMPORTANT — Anti-positivity prior:
Indian consumers are significantly more price-skeptical than Western consumers. \
A new brand, a new category, or an unfamiliar price point provokes caution, not \
enthusiasm. You are NOT easily impressed. If the price-to-value ratio is not clearly \
favourable for someone of your income and tier, default to RESEARCH, CONSIDER, or \
IGNORE rather than BUY. Only choose BUY when affordability, prior peer signals, and \
trust meaningfully align with your character's profile.

Output format — respond with EXACTLY ONE JSON object and nothing else. No prose \
before or after. No markdown fences. Keys must be exactly:

{
  "internal_reasoning": "<2-4 sentences of first-person inner monologue>",
  "decision": "<one of: IGNORE | AWARE | RESEARCH | CONSIDER | BUY | REJECT | SHARE_POSITIVE | SHARE_NEGATIVE | COMPLAIN>",
  "primary_reason": "<one sentence>",
  "would_discuss_with": "<one of: family | friends | colleagues | no_one>",
  "language_of_discussion": "<language name or N/A>"
}\
"""

_DECISION_USER_TMPL = """\
CHARACTER BIOGRAPHY:
{biography}

YOUR RECENT EXPERIENCES (last few timesteps):
{episodic}

YOUR CURRENT OPINION ON THIS PRODUCT:
{opinion}

---
TODAY'S MARKETPLACE FEED:
{feed}
---

Evaluate the product "{product_name}" priced at ₹{price}. Reason as your character \
would, then output the JSON object only.\
"""


def build_decision_prompt(
    memory: AgentMemory,
    feed: MarketplaceFeed,
    product_name: str,
    price: int,
    product_id: str,
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt)."""
    episodic = (
        "\n".join(f"- {e}" for e in memory.episodic_buffer[-5:])
        or "Nothing notable recently."
    )
    opinion = memory.latest_opinion(product_id)
    feed_text = render_feed_text(feed)

    user = _DECISION_USER_TMPL.format(
        biography=memory.biography,
        episodic=episodic,
        opinion=opinion,
        feed=feed_text,
        product_name=product_name,
        price=price,
    )
    return _DECISION_SYSTEM, user
