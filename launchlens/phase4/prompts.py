"""Decision prompt builder — converts agent context + feed into LLM input.

Output contract: the model is asked to return a single JSON object whose fields
match ``_REQUIRED_JSON_FIELDS`` in ``decisions.py``. We also accept the legacy
fielded-text format as a fallback (handled by the parser).

Embeds an *archetype-aware* price-skepticism prior aligned with research on
Indian consumer price sensitivity. The prior is advisory, not prescriptive —
it tells the model *how to weigh* evidence rather than what to default to,
because small instruction-following models otherwise park at the named default.

Enum values are written as separate quoted strings (no ``|`` separator) — a
3B-class quantized model will otherwise echo the ``|``-separated list as a
single value and trip the parser.
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

PRICE SKEPTICISM PRIOR:
Indian consumers weigh price-to-value carefully and are more skeptical than Western \
consumers. BUY when affordability AND value AND (peer signals OR your archetype's \
willingness to try new things) all align. REJECT clearly when the price-value ratio is \
poor, when the category doesn't fit your life, or when the brand is untrusted. Do NOT \
park indefinitely in RESEARCH or CONSIDER — choose one of those only if you genuinely \
need one more concrete piece of information.

ARCHETYPE GUIDANCE (be conservative — diffusion is slow in real markets):
- innovator: open to new things even without peer evidence, but ONLY when the price is \
  clearly comfortable for your income (well under 5% of monthly disposable). At the very \
  first week of a launch with no peer signals yet, even innovators are a tiny minority \
  (~2-3% of population) — most people, including most innovators, default to RESEARCH \
  or AWARE on week 1.
- early_adopter: BUY only after seeing at least ONE concrete BUY or SHARE_POSITIVE signal \
  from a trusted peer (family_elder or local_shopkeeper carry double weight). With no \
  peer signals, default to RESEARCH.
- early_majority: BUY only after MULTIPLE positive peer signals AND comfortable price.
- late_majority: BUY only after broad social proof (~30%+ of your peers have BUYed).
- laggard: REJECT or IGNORE unless the product is clearly indispensable for daily life.

Match your decision to YOUR archetype as stated in the biography below. If your feed \
shows ZERO peer purchases or positive shares, you almost certainly should NOT BUY this \
week unless you are an innovator with a price that is trivial for your income.

DECISION STATES (pick exactly one for the `decision` field):
IGNORE, AWARE, RESEARCH, CONSIDER, BUY, REJECT, SHARE_POSITIVE, SHARE_NEGATIVE, COMPLAIN.

DISCUSS TARGETS (pick exactly one for the `would_discuss_with` field):
family, friends, colleagues, no_one.

Output format — respond with EXACTLY ONE JSON object and nothing else. No prose \
before or after. No markdown fences.

{
  "internal_reasoning": "2-4 sentences of first-person inner monologue",
  "decision": "BUY",
  "primary_reason": "one sentence",
  "would_discuss_with": "friends",
  "language_of_discussion": "Hindi"
}

The values in the example above are illustrative only; pick your own. The keys \
must be exactly as shown.\
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
