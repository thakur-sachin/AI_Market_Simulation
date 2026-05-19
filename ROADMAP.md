# LaunchLens Roadmap — Deferred Work

Items that are **not** in the current build. Implementations called out below either remain unimplemented or are stubbed for forward-compatibility. None of the active code paths depend on them.

---

## Deferred Architectural Components

### LangGraph orchestration
**Status:** `langgraph>=0.2` is declared as a dependency but the actual loop in `phase4/loop.py` is a hand-rolled `asyncio.gather` batch. LangGraph adds stateful, checkpointable, graph-based coordination — useful at multi-thousand-agent scale but unnecessary at the 50–300 agent target on dev hardware.
**Triggers introduction:** runs > 1,000 agents AND a need to pause / resume mid-simulation.

### Semantic memory (pgvector / Postgres)
**Status:** `phase3/memory.py` only implements Tier-1 episodic storage (in-memory dict, optional Redis backend). No semantic embedding storage, no similarity retrieval. `opinion_embedding` was named in the original design but is not populated anywhere.
**Triggers introduction:** "find agents whose product opinion is semantically similar to X" becomes a routine query, OR > 20 timestep runs make context compression mandatory.

### Neo4j graph backend
**Status:** NetworkX is used for all graph construction. `phase2/graph_neo4j.py` does not exist.
**Triggers introduction:** > 10K agents across multiple districts; in-memory NetworkX traversals start hitting wall-clock limits.

### FastAPI / React dashboard
**Status:** `fastapi` and `uvicorn` are declared but no `launchlens/api/` package exists. The current UI is the Streamlit dashboard at `launchlens/gui/sim_dashboard.py`.
**Triggers introduction:** multi-tenant usage; async simulation runs that must survive client disconnects.

### NLP analytics (BERTopic, sentence-transformers)
**Status:** Both are declared as dependencies. `sentence-transformers` is used opportunistically inside `phase5/metrics.rejection_reason_alignment` (with a Jaccard fallback so CI runs without it). `bertopic` is **unused** — objection map / message resonance are not implemented.
**Files to add:** `phase6/nlp.py` (objection map, message resonance, feature priority), `phase6/segments.py` (K-means cluster characterization), `phase6/report.py` (Jinja2 + plotly + kaleido PDF).

---

## Calibration Data Collection (data work, not code work)

The three shipped calibration cases — `paper_boat_aam_panna.json`, `mamaearth_vitc.json`, `boat_airdopes_141.json` — currently carry **placeholder** adoption curves and top-segment / top-rejection lists. The metric machinery runs cleanly against them, but the gates are meaningless until real ground truth replaces the placeholders.

Sources to consult (per case):

| Product | Real-data sources |
|---|---|
| Paper Boat Aam Panna | AC Nielsen IRI panel data Q1-Q4 2018; Hector Beverages annual report; Mintel India 2018 packaged beverage tracker |
| mamaearth Vitamin C | Statista skincare 2021; Honasa Consumer earnings; Nykaa D2C skincare trends Q2/Q3 2021 |
| boAt Airdopes 141 | IDC India TWS tracker; aggregated Flipkart reviews 2022; Counterpoint Research TWS analysis 2022 |

Each case should fill:
- `real_adoption_curve` — cumulative BUY fraction per week for the first 8 weeks
- `real_top3_segments` — top-3 ISEC tiers among buyers
- `real_top3_rejections` — top-3 objection themes from review aggregation
- `category_reject_benchmark` — category-level REJECT rate from independent survey
- `source_citations` — replace `PLACEHOLDER` strings with concrete citations

---

## Real Demographic Data Ingestion

`data/raw/{census,nfhs,trai,nsso}/` are empty. `scripts/fetch_data_indore.py` currently produces a fully-fallback profile and prints download instructions. To replace `'fallback'` provenance with real source labels:

| File | Source | Notes |
|---|---|---|
| `data/raw/census/pca_district.csv` | censusindia.gov.in → Primary Census Abstract | Columns documented in `phase1/sources/census_pca.py` |
| `data/raw/census/c16_language.csv` | censusindia.gov.in → C-Series → C-16 Mother Tongue | Optional |
| `data/raw/nfhs/nfhs5_district.csv` | rchiips.org/nfhs/factsheet_NFHS-5 | Wealth quintiles + mobile-internet penetration |
| `data/raw/trai/trai_state_quarterly.csv` | trai.gov.in quarterly performance indicator reports | Manual extraction from PDFs |
| NSSO CES (state-level) | data.gov.in OGD API | Set `DATAGOVINDIA_API_KEY` + `NSSO_RESOURCE_ID` |

After placing files: `python scripts/fetch_data_indore.py` will rebuild the profile and report new provenance labels.

---

## Scaling Targets (post-API-keys, on cloud)

When migrating off the 8 GB VRAM laptop:

| Tier | Specs | Use |
|---|---|---|
| Local dev | RTX-class laptop GPU, 8 GB VRAM | 50–300 agents, mock + Ollama |
| Single-node cloud | EC2 `c6i.2xlarge` (8 vCPU, 16 GB), no GPU | 1K–5K agents, remote LLM |
| Multi-node | EC2 + ElastiCache (Redis) + RDS (Postgres + pgvector) | 5K–50K agents, multi-district |

Throughput target with remote providers: 50–100 agents/sec; one 5K-agent / 12-timestep run in ~10 minutes.

---

## Removed claims (formerly in README/CLAUDE.md)

The following claims appeared in earlier documentation but are not in the current code. They are listed here so reviewers can verify nothing is silently overstated:

- "Two-tier memory: Redis (episodic) + PostgreSQL + pgvector (semantic)" — only Tier-1 in-memory dict exists; Redis is optional and never enabled by default.
- "LangGraph orchestration (stateful, graph-based, checkpointable)" — replaced by a vanilla async loop.
- "FastAPI backend" — does not exist.
- "React + Recharts + Mapbox dashboard" — does not exist; UI is Streamlit.
- "BERTopic objection map" — not implemented.
- "<5% diversity validation" — was advertised but not enforced as a gate. Now available via `enforce_population_diversity()`; still not wired into the default CLI path (use `--strict` in future work).
- "datagovindia API integration" — present as a best-effort placeholder in `phase1/sources/nsso_datagovindia.py`; requires `DATAGOVINDIA_API_KEY` AND a stable `NSSO_RESOURCE_ID` env var to function.
