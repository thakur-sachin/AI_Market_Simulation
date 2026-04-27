"""Decision prompt builder — converts agent context + feed into LLM input."""
from __future__ import annotations

from launchlens.phase3.schemas import AgentMemory, MarketplaceFeed
from launchlens.phase3.feed import render_feed_text

_DECISION_SYSTEM = """\
You are role-playing as the person described below. Stay fully in character.
Do not break character or add meta-commentary.
Your responses must reflect this person's actual knowledge, biases, language patterns, and \
decision-making style.\
"""

_DECISION_USER_TMPL = """\
CHARACTER BIOGRAPHY:
{biography}

YOUR RECENT EXPERIENCES:
{episodic}

YOUR CURRENT OPINION ON THIS PRODUCT:
{opinion}

---
TODAY'S MARKETPLACE FEED:
{feed}
---

Based on your character, background, and what you have seen today, evaluate the product \
"{product_name}" priced at ₹{price}.

Respond in this EXACT format (do not add extra fields):

INTERNAL_REASONING: <2-4 sentences of internal monologue as your character>
DECISION: <exactly one of: IGNORE | AWARE | RESEARCH | CONSIDER | BUY | REJECT | SHARE_POSITIVE | SHARE_NEGATIVE | COMPLAIN>
PRIMARY_REASON: <one sentence>
WOULD_DISCUSS_WITH: <exactly one of: family | friends | colleagues | no_one>
LANGUAGE_OF_DISCUSSION: <language name or N/A>\
"""


def build_decision_prompt(
    memory: AgentMemory,
    feed: MarketplaceFeed,
    product_name: str,
    price: int,
    product_id: str,
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt)."""
    episodic = "\n".join(f"- {e}" for e in memory.episodic_buffer[-5:]) or "Nothing notable recently."
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
