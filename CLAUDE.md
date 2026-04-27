# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LaunchLens Labs — Synthetic Market Intelligence Engine. Simulates district-level Indian consumer behavior using an Agent-Based Model (ABM) powered by LLM personas. Target: <8% deviation from real-world adoption outcomes.

Methodological foundation: He et al. (2024) *Societies.io* paper (British Journal of Psychology) — proven that LLM agents natively reproduce homophily and community formation from persona prompts alone, without explicit social rules.

**Why ABM not GA:** A genetic algorithm converges to a homogeneous "super-buyer" population. ABM preserves heterogeneity — agents persist, information propagates (not fitness). Market simulation depends on that diversity.

---

## Architecture Overview (6 Phases → Pipeline)

```
Census/NFHS/NSSO/TRAI data
        ↓
[Phase 1] Bharat Diversity Engine
  - DistrictProfile JSON (766 districts)
  - Stratified persona sampling → Jinja2 → natural-language bios
  - Diversity validation: <5% deviation from source distributions
        ↓
[Phase 2] Social Graph
  - Watts-Strogatz small-world topology (k=6–10, β=0.1–0.3)
  - Agents ordered by similarity score before ring construction
  - Influencer node injection (Family Elder, Shopkeeper, Micro-Influencer, WA Hub)
  - NetworkX (<10K agents) or Neo4j (>10K agents)
        ↓
[Phase 3] Simulation Environment
  - ProductStimulus: structured JSON → natural language per agent
  - AgentMemory: Tier 1 (Redis, episodic, rolling 10-event buffer)
                  Tier 2 (PostgreSQL + pgvector, semantic, structured K/V)
  - MarketplaceFeed: personalized per agent from network connections
        ↓
[Phase 4] Interaction Loop  ← core simulation
  - LangGraph orchestration (stateful, graph-based, checkpointable)
  - Decision prompt: biography + episodic buffer + product opinion + feed
  - 9 outputs: IGNORE / AWARE / RESEARCH / CONSIDER / BUY / REJECT /
               SHARE_POSITIVE / SHARE_NEGATIVE / COMPLAIN
  - Network propagation: social actions fan-out to direct connections
  - Cascade attenuation: 30% salience reduction per hop
  - Timestep ≈ 1 real week; runs = 12–24 timesteps
        ↓
[Phase 5] Validation & Calibration  ← critical loop
  (see section below)
        ↓
[Phase 6] Output & Analytics
  - 8 deliverables auto-generated from simulation logs
  - FastAPI backend + React/Recharts/Mapbox dashboard
```

---

## Validation Loop (Priority Concern)

This is the mechanism that keeps the system honest against real-world outcomes.

**Calibration dataset:** 10–15 historical Indian product launches with known outcomes (adoption curves, early-adopter demographics, marketing channels). Starting set: Paper Boat, mamaearth, Zoho, Raw Pressery, boAt.

**5 validation metrics per calibration run:**

| Metric | Method | Gate |
|---|---|---|
| Overall Adoption Rate | % agents reaching BUY vs real data | <8% deviation |
| Adoption Curve Shape | Dynamic Time Warping (DTW) | <0.15 normalized DTW |
| Top Segment Accuracy | Top 3 demographic segments match | ≥2 of 3 match |
| Regional Heat Map | Spearman rank correlation | ρ > 0.7 |
| Rejection Reason Alignment | Top 3 rejection reasons vs survey | ≥2 of 3 match |

**Tuning signals:**
- Adoption rate too high → increase price sensitivity weight in persona prompts
- Adoption rate too low → increase social proof multiplier
- Wrong segments adopting → audit persona generation for demographic accuracy
- Curve too fast → reduce propagation speed / increase timestep duration

**Bias checks (run before calibration):**
1. **Affluence bias** — compare SEC-D/E agent behavior against low-income consumer research
2. **Positivity bias** — compare simulated rejection rates against known category rejection rates
3. **Language bias** — validate regional-language reasoning against human raters
4. **Homogeneity bias** — measure response variance within same-demographic cohorts

**Cross-validation:** Hold out 3 products as test set; calibrate on remainder; measure test-set performance.

**Persona QA (per-generation):** 5 sanity-check prompts per persona; flag for regeneration if ≥2 fail consistency.

---

## LLM Routing

| Agent Profile | Primary | Fallback |
|---|---|---|
| English-dominant, SEC A/B, Urban | Claude Sonnet 4 | GPT-4o-mini |
| Hindi/Hinglish, any SEC | Sarvam-105B | Sarvam-M (24B) |
| Regional language (Tamil, Telugu, Bengali, Marathi, etc.) | Sarvam-105B | BharatGen Param 2 |
| Rural, SEC D/E, low literacy | Sarvam-105B + low-income prompt | Sarvam-M constrained |
| Persona generation (batch) | Claude Sonnet 4 | Sarvam-M |
| Post-hoc reasoning analysis | Claude Sonnet 4 | — |

Use **prompt engineering, not fine-tuning** for persona instantiation (Toubia et al. 2025; Arora et al. 2025 — prompt-augmented LLMs outperform fine-tuned models for behavioral prediction).

Cost target: ~$95–$190 per 5K-agent, 12-timestep run (60K LLM calls).

---

## Key Data Contracts

**`DistrictProfile`** (JSON, one per district):
- `population_count`, `age_distribution` (5-year buckets), `sex_ratio`, `urban_rural_split`
- `literacy_rate`, `language_distribution` (Census C-16 mother tongue tables)
- `isec_nccs_distribution` (12-tier A1–E3), `smartphone_penetration`, `median_monthly_expenditure`

**`AgentMemory`** (per agent, stored across timesteps):
- `biography: str` — immutable persona text
- `episodic_buffer: List[str]` — last 10 events (Redis, TTL-based)
- `product_awareness: Dict[str, float]` — product_id → 0–1
- `product_opinion: Dict[str, str]` — product_id → natural language
- `purchase_history: List[dict]` — timestep, product_id, decision, reasoning
- `opinion_embedding: np.array` — pgvector-stored for semantic retrieval

**`AgentDecision`** (structured output, parsed from LLM):
- `internal_reasoning: str` — the "thought" (source for Message Resonance and Objection Map analytics)
- `decision: Literal[IGNORE|AWARE|RESEARCH|CONSIDER|BUY|REJECT|SHARE_POSITIVE|SHARE_NEGATIVE|COMPLAIN]`
- `primary_reason: str`
- `would_discuss_with: Literal[family|friends|colleagues|no_one]`
- `language_of_discussion: str`

**`ProductStimulus`** (JSON, client-provided):
- `product_name`, `category`, `price` (mrp + launch_offer + currency)
- `key_features: List[str]`, `distribution: List[str]`
- `marketing_copy`, `competitor_context`, `target_segment`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data pipeline | Python, pandas, geopandas, `datagovindia` PyPI package |
| Persona templates | Jinja2 |
| Graph (small) | NetworkX (in-memory, <10K agents) |
| Graph (large) | Neo4j Aura Professional |
| Episodic memory | Redis (AWS ElastiCache) |
| Semantic memory | PostgreSQL + pgvector (AWS RDS) — prefer over Pinecone/Milvus until >10K concurrent agents |
| Orchestration | LangGraph (stateful multi-agent, MIT license) |
| NLP analytics | BERTopic (objection map), scikit-learn K-means (segment clustering), sentence-transformers |
| Backend API | FastAPI |
| Dashboard | React + Recharts + Mapbox |
| Infra | AWS EC2 c6i.2xlarge, Vercel/Amplify (dashboard) |

---

## Simulation Parameters

- **Population per run:** 1,000–5,000 agents (start with 100-agent tests)
- **Timestep:** ~1 real week
- **Run duration:** 12–24 timesteps (3–6 months of simulated market activity)
- **Launch phase:** first 2–4 timesteps
- **Social graph degree (k):** 6–10 (product-relevant social circle, not total connections)
- **Rewiring probability (β):** 0.1–0.3
- **Feed composition per agent per timestep:** 1 product ad + up to 5 peer reviews + up to 3 peer purchase decisions + 1–2 competitor mentions + 1 market noise item
- **Propagation salience decay:** 30% per hop
- **Batch size for LLM calls:** 100–500 agents async; target 50–100 agents/second

---

## Output Deliverables → Source Data

| Deliverable | Source | Method |
|---|---|---|
| Market Fit | Decision distribution (BUY/REJECT/IGNORE) | Aggregate by demographic segment |
| City Intelligence | Decisions grouped by district | Geographic heat map |
| Adoption Curve | Cumulative BUY per timestep | Time-series with CI from multiple runs |
| Segment Depth | Decision vectors by agent | K-means clustering |
| Message Resonance | `internal_reasoning` logs | NLP sentiment per marketing claim |
| Feature Priority | Feature mentions in BUY vs REJECT reasoning | Feature importance score |
| Objection Map | REJECT + COMPLAIN reasoning | BERTopic topic modeling |
| Scenario Analysis | Multiple runs with varied parameters | Delta comparison tables |

---

## Development Sequence (18-week MVP)

1. **Wks 1–3:** Demographic pipeline + first 1,000 personas for Indore (gate: diversity check passes)
2. **Wks 4–5:** Social graph + influencer nodes (gate: small-world metrics match benchmarks)
3. **Wks 6–8:** Memory system + feed + product stimulus schema (gate: single-agent end-to-end produces valid `AgentDecision`)
4. **Wks 9–11:** Full interaction loop, 100-agent test (gate: observable network cascades in 6 timesteps)
5. **Wks 12–14:** Calibration against 3 historical products (gate: ≥1 product <10% deviation)
6. **Wks 15–17:** Analytics pipeline + dashboard MVP (gate: all 8 deliverables auto-generated)
7. **Wk 18:** First 5,000-agent run + client-ready report

---

## Known Risks

- **Homogeneity:** LLMs tend toward similar outputs within same-demographic cohorts. Mitigate with ±15% stochastic variation in persona prompts and temperature tuning.
- **Positivity bias:** LLMs over-generate BUY. Add skepticism prior explicitly in prompts (Indian consumers are more price-skeptical than Western consumers per research).
- **Census staleness:** Census 2011 data. Cross-reference with NFHS-5 (2021) and TRAI quarterly data.
- **Character breaks:** Agents produce generic LLM responses rather than staying in persona. Enforce with explicit stay-in-character instructions and post-hoc filtering.
- **Calibration floor:** v1 target is <10% deviation; tighten to <8% post-MVP.
