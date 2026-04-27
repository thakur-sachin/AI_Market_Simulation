# LaunchLens — Next Steps for Full-Scale System

This document outlines the work needed to graduate from the current 100-agent, mock-LLM prototype to a production-grade synthetic market intelligence system. Each section maps to a phase in the architecture and is ordered by implementation priority.

---

## Phase 5 — Validation & Calibration Loop  ⚠️ CRITICAL PATH

The mechanism that makes the system commercially defensible.

### 5.1 Calibration Dataset
Build ground-truth cases from real Indian product launches:

| Product | Category | Key Metric Source |
|---|---|---|
| Paper Boat Aam Panna | Beverages | AC Nielsen IRI panel data |
| mamaearth Vitamin C Serum | Skincare | Statista + earnings disclosures |
| boAt Airdopes 141 | Consumer Electronics | IDC India, Flipkart reviews |
| Zoho One | B2B SaaS | Case studies (public) |
| Raw Pressery | Cold-press juice | Funding announcement + media |

For each: collect real adoption curve (quarterly units sold or downloads), top-adopter demographics, top rejection reasons (from reviews), regional distribution.

### 5.2 Five-Metric Validation Protocol
Implement `launchlens/phase5/metrics.py`:

```python
def adoption_rate_deviation(sim_rate: float, real_rate: float) -> float: ...
# Gate: < 8%

def dtw_curve_distance(sim_curve: list[float], real_curve: list[float]) -> float: ...
# Gate: normalized DTW < 0.15

def top_segment_accuracy(sim_top3: list[str], real_top3: list[str]) -> int: ...
# Gate: >= 2 of 3 match

def regional_spearman(sim_district_rates: dict, real_district_rates: dict) -> float: ...
# Gate: rho > 0.70

def rejection_reason_alignment(sim_reasons: list[str], real_reasons: list[str]) -> int: ...
# Uses BERTopic + cosine similarity to match topics. Gate: >= 2 of 3 match
```

### 5.3 Bias Detection Suite
Run before every calibration cycle:

- **Affluence bias**: Compare mean p(BUY) for D/E tier agents vs. published low-income consumer research. Flag if > 20% deviation.
- **Positivity bias**: Compare simulated REJECT rate to category-level rejection benchmarks. LLMs over-generate BUY.
- **Language bias**: Sample 50 Hindi/regional-language `internal_reasoning` outputs; human-rate for cultural authenticity (1–5 scale). Gate: mean ≥ 3.5.
- **Homogeneity bias**: Measure response variance within same ISEC cohort. Gate: Gini coefficient of decisions > 0.3 (ensuring diversity).

### 5.4 Tuning Signals
After calibration, apply parameter adjustments:

| Failure Mode | Tuning Action |
|---|---|
| Adoption rate too high | Increase `price_sensitivity_weight` in persona prompts; decrease `_ISEC_BASE_BUY` for lower tiers |
| Adoption rate too low | Increase `social_proof_multiplier` in `_mock_decision` / decision prompt |
| Wrong segments buying | Audit `sample_demographic_vectors` — check ISEC distribution matches district profile |
| Curve too fast | Reduce `_ARCHETYPE_SPEED` multipliers; increase social signal threshold |
| Wrong regional spread | Audit `_disaggregate_isec` for state-level adjustments |

### 5.5 Cross-Validation
- Hold out 3 products as test set
- Calibrate on remaining 7–12
- Report test-set performance for investor/client credibility

---

## Phase 4 Upgrades — Real LLM Integration

Replace the mock decision engine with actual LLM calls. The scaffolding already exists in `phase4/loop.py`.

### 4.1 Sarvam Integration
```bash
# .env
SARVAM_API_KEY=...
SARVAM_BASE_URL=https://api.sarvam.ai/v1
```
Sarvam-105B is OpenAI-compatible; `launchlens/llm.py` already routes to it. Test with `python -m launchlens.cli generate-personas --district Indore --n 10`.

### 4.2 Persona QA Pipeline
`launchlens/phase1/persona_qa.py` is written but not wired into the CLI. Enable it:
```bash
python -m launchlens.cli generate-personas --district Indore --n 100 --qa
```
Regenerate any persona with ≥ 2/5 QA failures.

### 4.3 Anti-Positivity Prompt Engineering
Add skepticism prior to `phase4/prompts.py`:
> "Indian consumers are significantly more price-skeptical than Western consumers. You are not easily impressed. If the price-to-value ratio is not clear, default to RESEARCH or IGNORE."

### 4.4 Cost Guardrails
- Track tokens per run; emit warning if projected cost > $200
- Add `--dry-run` flag that builds everything but skips LLM calls (uses mock decisions)
- Cache LLM responses for identical (biography, product, memory) triples — saves ~30% on repeat runs

---

## Phase 3 Upgrades — Production Memory

### 3.1 Redis Episodic Memory
```bash
# Local dev
docker run -d -p 6379:6379 redis:7-alpine

# .env
REDIS_URL=redis://localhost:6379/0
```
`launchlens/phase3/memory.py` already implements `RedisMemoryBackend` with TTL. Enable by setting `REDIS_URL`.

### 3.2 PostgreSQL + pgvector Semantic Memory
Wire up `phase3/memory.py` semantic layer:
- Store `opinion_embedding` (sentence-transformers, `all-MiniLM-L6-v2`) in pgvector
- Enable semantic peer retrieval: "find 3 agents with similar opinion on this product category"
- Schema: `agent_memories(agent_id TEXT PK, biography TEXT, episodic_json JSONB, opinion_vector VECTOR(384))`

### 3.3 Memory Compression
Rolling 10-event episodic buffer is fine for <20 timesteps. For longer runs:
- Add summarization step every 10 events: LLM compresses old events into a paragraph
- Keep last 5 raw events + summary

---

## Phase 2 Upgrades — Graph at Scale

### 2.1 Neo4j for > 10K Agents
```python
# launchlens/phase2/graph_neo4j.py
def build_graph_neo4j(personas, k=8, beta=0.15, neo4j_uri=...) -> SimGraph:
    # MERGE nodes, CREATE edges in batches of 1000
    # Use APOC for Watts-Strogatz rewiring
```
Gate: > 10K agents AND multi-district runs.

### 2.2 Multi-District Graphs
Current graphs are single-district. For city-level simulations:
- Build per-district subgraphs with local k=8
- Add cross-district edges (`add_cross_district_edges` already written) with edge fraction ≈ 0.01–0.03
- Weight cross-district edges by transport connectivity (road distance, train frequency)

### 2.3 Dynamic Graph Evolution
Real social networks change over time:
- Each timestep: with probability 0.02 per agent, rewire one edge (simulate new social connections)
- Post-BUY: increase connection probability to other BUY agents (social proof clustering)

---

## Phase 6 — Analytics & Output Pipeline

Eight deliverables to auto-generate from `SimulationLog`.

### 6.1 NLP Analytics  (`launchlens/phase6/nlp.py`)
```python
# Message Resonance — which marketing claims appear in BUY reasoning?
def extract_resonant_claims(decisions: list[AgentDecision], marketing_copy: str) -> dict[str, float]:
    # Embed each claim + each internal_reasoning; cosine similarity > 0.6 = resonance

# Objection Map — what reasons appear in REJECT/COMPLAIN reasoning?
def build_objection_map(decisions: list[AgentDecision]) -> list[dict]:
    # BERTopic on primary_reason text; return top 5 topics with representative quotes

# Feature Priority — which features mentioned most in BUY vs REJECT?
def feature_importance(decisions: list[AgentDecision], features: list[str]) -> dict[str, float]:
    # Keyword frequency in BUY reasoning / REJECT reasoning → delta score
```

### 6.2 Segment Clustering (`launchlens/phase6/segments.py`)
```python
def cluster_adopters(decisions: list[AgentDecision], personas: list[AgentPersona]) -> list[dict]:
    # Embed agent demographic vectors → K-means (k=4–6) → characterize each cluster
    # Output: cluster label, size, top ISEC tiers, top archetypes, adoption rate
```

### 6.3 Report Generation (`launchlens/phase6/report.py`)
Auto-generate PDF/HTML report from all analytics:
- Use `jinja2` for HTML template (already a dependency)
- Embed plotly charts as static PNG (use `kaleido`)
- Sections: Executive Summary, Market Fit Score, Adoption Curve, Top Segments, Message Resonance, Objection Map, Scenario Comparison

### 6.4 FastAPI Backend (`launchlens/api/`)
```
POST /simulations/run     — submit sim job (async, returns job_id)
GET  /simulations/{id}    — poll status / retrieve SimulationLog
GET  /simulations/{id}/report  — download generated report
GET  /districts           — list available DistrictProfiles
POST /products/validate   — validate ProductStimulus schema
```
Use `BackgroundTasks` for async simulation runs. Store results in PostgreSQL.

### 6.5 Dashboard Upgrade (`launchlens/gui/`)
Current Streamlit dashboard covers simulation viz. Production dashboard needs:
- **Map view**: Mapbox choropleth of adoption rate by district (requires multi-district runs)
- **Scenario comparison**: Run A/B simulations (e.g. price ₹79 vs ₹99) and diff the outputs
- **Live run monitoring**: WebSocket updates from FastAPI backend during simulation
- **Report download button**: Trigger `phase6/report.py` and return PDF

---

## Infrastructure & DevOps

### Local Development
```bash
# Start all local dependencies
docker-compose up -d   # Redis + PostgreSQL + pgvector

# Run full pipeline test (100 agents, real LLMs)
python -m launchlens.cli generate-personas --district Indore --n 100 --seed 42
python -m launchlens.sim_lite --agents 100 --timesteps 8

# Dashboard
streamlit run launchlens/gui/sim_dashboard.py
```

### Production (AWS)
- **EC2**: c6i.2xlarge (8 vCPU, 16GB RAM) — handles 5K-agent runs
- **ElastiCache**: Redis for episodic memory (r7g.large)
- **RDS**: PostgreSQL 16 + pgvector extension (db.r7g.large)
- **S3**: Store `DistrictProfile` JSONs, simulation logs, generated reports
- **Target throughput**: 50–100 agents/second → 5K-agent, 12-timestep run in ~10 min

### Testing Gaps (fill before scaling)
- `tests/test_phase3/` — MemoryStore CRUD, feed construction, peer signal dedup
- `tests/test_phase4/` — decision parser edge cases, propagation signal counts, adoption curve shape
- `tests/test_phase5/` — metric calculation correctness against known fixtures
- Integration test: end-to-end 50-agent run with mock LLM; assert `sim_log.adoption_curve()[-1] > 0`

---

## 18-Week Milestone Gates (remaining)

| Week | Gate |
|---|---|
| **9–11** | Full interaction loop with real LLMs, 100-agent test — observable network cascades in 6 timesteps |
| **12–14** | Calibration against 3 historical products — ≥ 1 product < 10% deviation |
| **15–17** | All 8 analytics deliverables auto-generating from SimulationLog |
| **18** | First 5,000-agent run + client-ready report |

---

## Immediate Next Actions (in priority order)

1. **Add `pydantic-settings` to `pyproject.toml`** ✅ Done
2. **Wire Sarvam API key** — single agent end-to-end sanity check before batch runs
3. **Enable persona QA** — `run_qa_batch` in CLI pipeline
4. **Build `phase5/metrics.py`** — adoption rate deviation + DTW (need `dtaidistance`)
5. **Collect 3 calibration products** — Paper Boat, mamaearth, boAt with real adoption data
6. **Phase 3 tests** — MemoryStore + feed builder coverage before touching production memory
7. **FastAPI stub** — `POST /simulations/run` with background task + polling
