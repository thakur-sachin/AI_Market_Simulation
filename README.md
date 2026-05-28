# LaunchLens Labs

**Synthetic Market Intelligence Engine for Indian Consumer Markets**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Phase](https://img.shields.io/badge/phase-1--4%20complete-brightgreen.svg)](#project-status)
[![Validation Target](https://img.shields.io/badge/target-<%208%25%20adoption%20deviation-orange.svg)](#phase-5--validation--calibration)
[![Methodology](https://img.shields.io/badge/methodology-He%20et%20al.%202024-purple.svg)](#research-foundation)

---

LaunchLens simulates how real Indian consumers discover, evaluate, and decide on new products — before those products exist in the market. It builds a synthetic population of 1,000–5,000 LLM-powered agents drawn from actual Census, NFHS, and TRAI data, connects them in a realistic social network, and runs a week-by-week diffusion simulation. The output is not a survey or a focus group — it is a living, propagating social system that surfaces adoption curves, segment depth, message resonance, and objection maps.

**Target accuracy:** <8% deviation from real-world adoption outcomes, validated against historical Indian product launches.

---

## How It Works

```
Census / NFHS-5 / TRAI data
           │
           ▼
┌─────────────────────────────┐
│  Phase 1 · Bharat Diversity │  766 DistrictProfiles · stratified persona sampling
│         Engine              │  Jinja2 biographies · diversity validation <5% drift
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 2 · Social Graph     │  Watts-Strogatz small-world (k=6-10, β=0.1-0.3)
│                             │  Homophily ordering · 4 influencer archetypes
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 3 · Sim Environment  │  ProductStimulus → per-agent feed
│                             │  AgentMemory: Redis (episodic) + pgvector (semantic)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 4 · Interaction Loop │  LLM decision per agent per timestep (≈1 week)
│           ← core ─          │  9 decision states · 30% salience decay per hop
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 5 · Validation       │  5-metric calibration loop against real launches
│           [planned]         │  Bias detection · DTW curve matching
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 6 · Analytics        │  8 deliverables: adoption curve, objection map,
│           [planned]         │  segment depth, message resonance, FastAPI + React
└─────────────────────────────┘
```

---

## Quick Start

### Option 1 — No infrastructure, no API keys (sim_lite)

Runs a complete 100-agent simulation using a stochastic mock decision engine. No Redis, no pgvector, no LLM calls needed.

```bash
git clone <repo>
cd LaunchLensLab
pip install -e ".[dev]"

python -m launchlens.sim_lite --agents 100 --timesteps 8 --seed 42
```

Output: colored terminal bar charts showing decision distribution, adoption curve, and consensus entropy per timestep.

```
  Timestep  6  │  Agents: 100  │  Cumulative adoption: 3.0%
  ────────────────────────────────────────────────────────
  IGNORE           ████████████████████████░░░░░░    80 (80.0%)
  AWARE            ███░░░░░░░░░░░░░░░░░░░░░░░░░░░    10 (10.0%)
  RESEARCH         █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░     4 ( 4.0%)
  CONSIDER         ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░     3 ( 3.0%)
  BUY              ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░     3 ( 3.0%)
```

### Option 2 — Interactive Streamlit Dashboard

```bash
streamlit run launchlens/gui/sim_dashboard.py
# → http://localhost:8501
```

Adjust agents (50–500), timesteps, seed, and product parameters from the sidebar. Three tabs:
- **Overview** — stacked decision-state area chart + adoption curve / entropy evolution
- **Network** — live social graph with timestep scrubber; nodes colored by decision state, influencer nodes enlarged
- **Segments** — ISEC × state heatmap, adoption rate by tech archetype, propagation signal chart

### Option 3 — Local LLM validation (free, no API keys)

For feasibility testing without API costs, run any small open-weight model locally via Ollama. The router transparently sends every Claude/Sarvam call to the local endpoint.

```bash
# 1. Install Ollama (Linux/macOS/Windows — https://ollama.com)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull a tiny model (pick one based on your hardware)
ollama pull qwen2.5:0.5b      # ~400 MB, fastest, CPU-friendly
ollama pull llama3.2:1b       # ~1.3 GB, better English coherence
ollama pull gemma2:2b         # ~1.6 GB, strongest small option
# (Optional, larger if you have ≥6GB RAM/VRAM:)
# ollama pull qwen2.5:3b       # ~2 GB, much better instruction following

# 3. Verify it's serving (Ollama auto-starts a server on :11434)
curl http://localhost:11434/api/tags

# 4. Configure LaunchLens
cp .env.example .env
# In .env, set:
#   LOCAL_LLM_ENABLED=true
#   LOCAL_LLM_MODEL=qwen2.5:0.5b

# 5. Generate a few personas (LLM-driven biographies)
python -m launchlens.cli generate-personas --district Indore --n 5 --seed 42 --local --skip-qa

# 6. Run the full LLM-driven simulation loop
python -m launchlens.cli run-sim --agents 20 --timesteps 5 --seed 42 --local

# 7. Run with calibration against a built-in fixture
python -m launchlens.cli run-sim --agents 30 --timesteps 8 --seed 42 --local \
  --fixture launchlens/phase5/fixtures/paper_boat_aam_panna.json
```

Outputs land in `data/processed/sim_logs/`:
- `prod_001_sim.json` — full SimulationLog with per-timestep decisions
- `prod_001_report.md` — 8-deliverable markdown report (adoption curve, segments, objection map, message resonance, validation gates)

**Hardware notes:**
- `qwen2.5:0.5b` runs comfortably on CPU; 20 agents × 5 timesteps ≈ 5–15 minutes
- On Apple Silicon, GPU acceleration is automatic via Metal
- Decision parsing tolerates the messier output of tiny models (lenient + fuzzy fallback)

### Option 4 — Full pipeline with hosted LLMs (Claude + Sarvam)

```bash
cp .env.example .env
# Fill ANTHROPIC_API_KEY and SARVAM_API_KEY

# Build district profiles from Census data
python -m launchlens.cli ingest-census

# Generate 1,000 personas for Indore with QA
python -m launchlens.cli generate-personas --district Indore --n 1000 --seed 42

# Run full simulation (Phase 4 loop.py — requires Redis + PostgreSQL for scale)
python -m launchlens.cli run-sim --district Indore --agents 1000 --timesteps 12
```

---

## Architecture Reference

### Phase 1 · Bharat Diversity Engine

| Module | Purpose |
|---|---|
| `phase1/schemas.py` | `DistrictProfile`, `DemographicVector`, `AgentPersona` Pydantic models |
| `phase1/data_pipeline.py` | Builds district profiles from Census 2011 + NFHS-5 cross-reference |
| `phase1/persona_gen.py` | Stratified sampler → Jinja2 biographies via LLM; 9 language name banks |
| `phase1/persona_qa.py` | 5-prompt QA gate per persona; regenerates if ≥2/5 fail |
| `templates/persona_bio.j2` | Biography template: daily routine, media habits, shopping behaviour, money attitude |

**Key design choices:**
- Agents are sampled adult-only (15+) with ±15% stochastic income variation within ISEC tier
- ISEC/NCCS 12-tier classification (A1–E3) drives price sensitivity, LLM routing, and social influence
- `validate_population_diversity()` checks urban/rural, ISEC, language marginals to within <5% of source data

### Phase 2 · Social Graph

| Module | Purpose |
|---|---|
| `phase2/graph.py` | Watts-Strogatz builder with homophily pre-ordering; small-world validation |
| `phase2/influencers.py` | Injects 4 influencer archetypes; rewires edges to target degree ranges |
| `phase2/schemas.py` | `SimGraph`, `NodeMeta`, `InfluencerArchetype` |

**Homophily without explicit rules:** agents are sorted by `(district, ISEC band, age band, language)` before ring construction. Demographically similar agents become k-nearest neighbors — emergent community structure, no special rules needed. This replicates the He et al. (2024) finding that LLM agents reproduce homophily from persona prompts alone.

**Influencer archetypes:**

| Archetype | Share | Degree | Awareness mult | Trust mult |
|---|---|---|---|---|
| Family Elder | 10% | 3–5 | 1.0× | **2.0×** |
| Local Shopkeeper | 3% | 15–25 | **1.5×** | 1.0× |
| Micro-Influencer | 0.8% | 50–200 | 1.3× | 0.8× |
| WhatsApp Hub | 6.5% | 10–15 | 1.2× | 1.0× |

### Phase 3 · Simulation Environment

| Module | Purpose |
|---|---|
| `phase3/schemas.py` | `ProductStimulus`, `AgentMemory`, `PeerSignal`, `AgentDecision`, `MarketplaceFeed` |
| `phase3/memory.py` | `MemoryStore` with `InMemoryBackend` (always on) and `RedisMemoryBackend` (production) |
| `phase3/feed.py` | Builds personalized per-agent feed; deduplicates signals, caps at 5 reviews / 3 purchases |

**Memory architecture:**
- **Tier 1 (episodic):** rolling 10-event buffer per agent — Redis in production, in-memory for dev
- **Tier 2 (semantic):** `product_opinion` text + pgvector embedding for similarity retrieval

**9 Decision States:**

```
IGNORE → AWARE → RESEARCH → CONSIDER → BUY
                                  ↓
                    SHARE_POSITIVE / COMPLAIN
          REJECT ←────── (at any point)
                    SHARE_NEGATIVE
```

States that propagate to the social network: `BUY`, `SHARE_POSITIVE`, `SHARE_NEGATIVE`, `COMPLAIN`

### Phase 4 · Interaction Loop

| Module | Purpose |
|---|---|
| `phase4/loop.py` | Main async simulation loop; batched LLM calls; `SimulationLog`, `TimestepLog` |
| `phase4/prompts.py` | Decision prompt: biography + episodic buffer + product opinion + feed |
| `phase4/decisions.py` | Parses structured `AgentDecision` from LLM output; fuzzy fallback for malformed responses |
| `phase4/propagation.py` | Social signal fan-out to direct neighbors; 30% salience decay per hop |
| `sim_lite.py` | Self-contained mock simulation — no external infrastructure required |

**Propagation mechanics:**
- Each `PROPAGATING_STATE` decision fans out to all direct network neighbors
- `base_salience = archetype_multiplier × complain_boost (1.5× for complaints)`
- Signals decay 30% per timestep; pruned below 0.05 salience floor
- Family Elder trust signals carry 2× weight in BUY probability calculation

---

## LLM Routing

The dual-model routing is critical for demographic authenticity. Western LLMs systematically misrepresent lower-income, rural, and regional-language consumers.

| Agent Profile | Primary Model | Fallback |
|---|---|---|
| English, SEC A/B, urban | Claude Sonnet 4.6 | GPT-4o-mini |
| Hindi / Hinglish, any SEC | Sarvam-105B | Sarvam-M (24B) |
| Regional language (Tamil, Telugu, Bengali, etc.) | Sarvam-105B | BharatGen Param 2 |
| Rural, SEC D/E, low literacy | Sarvam-105B + constrained prompt | Sarvam-M |

Sarvam-105B uses an OpenAI-compatible endpoint — `launchlens/llm.py` routes transparently via `LLMRoute.CLAUDE` or `LLMRoute.SARVAM`.

**Cost target:** ~$95–$190 per 5,000-agent, 12-timestep run (≈60,000 LLM calls).

---

## Key Data Contracts

<details>
<summary><strong>DistrictProfile</strong> — one JSON per district, 766 total</summary>

```python
DistrictProfile(
    district_id="MP001",
    district_name="Indore",
    population=3_276_697,
    age_distribution=AgeDistribution(...),   # 5-year buckets, sums to 1
    sex_ratio=920,                           # females per 1000 males
    urban_share=0.70,
    literacy_rate=0.82,
    language_distribution={"hindi": 0.85, "urdu": 0.08, "english": 0.07},
    isec_distribution={"A1": 0.03, ..., "E3": 0.04},   # sums to 1
    smartphone_penetration=0.62,             # derived from TRAI + NFHS
    internet_penetration=0.50,
    upi_adoption=0.38,
)
```
</details>

<details>
<summary><strong>ProductStimulus</strong> — client-provided product brief</summary>

```python
ProductStimulus(
    product_id="prod_001",
    product_name="FreshBite Protein Bar",
    category="Health & Nutrition",
    price_mrp=99, price_launch=79,
    key_features=["20g protein", "No added sugar", "Mango / Chocolate flavors"],
    distribution_channels=["Amazon India", "BigBasket", "Modern Trade"],
    marketing_copy="Fuel your grind. India's first truly tasty protein bar.",
    competitor_context="Yoga Bar (₹50-80), RiteBite Max (₹80-120)",
    target_segment="Health-conscious urban millennials, 22-35, SEC A/B",
)
```
</details>

<details>
<summary><strong>AgentDecision</strong> — structured LLM output per agent per timestep</summary>

```python
AgentDecision(
    agent_id="agent_0042",
    product_id="prod_001",
    timestep=5,
    internal_reasoning="Given my income of ₹28,000...",  # source for message resonance analytics
    decision="CONSIDER",                                  # one of 9 states
    primary_reason="Comparing with Yoga Bar before committing.",
    would_discuss_with="friends",                         # family | friends | colleagues | no_one
    language_of_discussion="hindi",
)
```
</details>

---

## Configuration

```bash
cp .env.example .env
```

```env
# LLM APIs
ANTHROPIC_API_KEY=sk-ant-...
SARVAM_API_KEY=...
SARVAM_BASE_URL=https://api.sarvam.ai/v1

# Storage (optional for dev — falls back to in-memory)
POSTGRES_URL=postgresql://user:pass@localhost:5432/launchlens
REDIS_URL=redis://localhost:6379/0

# Simulation
DEFAULT_AGENT_COUNT=1000
LLM_BATCH_SIZE=200
LLM_MAX_CONCURRENT=50
```

All settings are validated on startup via `pydantic-settings`. If `REDIS_URL` is unset, `MemoryStore` falls back silently to the in-memory backend.

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest                    # all tests
pytest tests/test_phase1  # phase 1 only
pytest -v -k "graph"      # by keyword
```

**Current coverage:**

| Suite | Tests | Status |
|---|---|---|
| `tests/test_phase1/test_schemas.py` | 9 tests — schema validation, ISEC disaggregation, stratified sampling, diversity, income bounds | ✅ All pass |
| `tests/test_phase2/test_graph.py` | 11 tests — node count, connectivity, degree, small-world σ, homophily, influencer proportions, serialization | ✅ All pass |
| `tests/test_phase3/` | 9 tests — MemoryStore CRUD, episodic buffer, feed dedup, archetype_hint resolution | ✅ All pass |
| `tests/test_phase4/` | 13 tests — decision parser (strict/lenient/fuzzy), propagation decay, signal replacement, archetype multipliers | ✅ All pass |
| `tests/test_phase5/` | 12 tests — adoption deviation, DTW shape match, segment accuracy, Spearman, reject alignment | ✅ All pass |

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| **1** | Bharat Diversity Engine — district profiles + persona generation | ✅ Complete |
| **2** | Social Graph — small-world construction + influencer injection | ✅ Complete |
| **3** | Simulation Environment — memory, feed, product stimulus | ✅ Complete |
| **4** | Interaction Loop — LLM decisions, propagation, simulation log | ✅ Complete |
| **4b** | `sim_lite` — zero-infra mock simulation + Streamlit dashboard | ✅ Complete |
| **5** | Validation & Calibration — 5-metric loop + 3-check bias suite | ✅ Complete |
| **6** | Analytics & Output — objection map, feature priority, message resonance, segment depth, markdown report | ✅ Complete (FastAPI/React deferred) |

See [`NEXT_STEPS.md`](NEXT_STEPS.md) for the full implementation roadmap with prioritized actions, infra setup, and 18-week milestone gates.

---

## Project Structure

```
LaunchLensLab/
├── launchlens/
│   ├── config.py              # pydantic-settings; get_settings() with @lru_cache
│   ├── llm.py                 # LLMRoute enum + unified complete() + routing logic
│   ├── cli.py                 # ingest-census, generate-personas commands
│   ├── sim_lite.py            # self-contained mock simulation (no infra needed)
│   ├── phase1/
│   │   ├── schemas.py         # DistrictProfile, DemographicVector, AgentPersona
│   │   ├── data_pipeline.py   # Census + NFHS data ingestion and disaggregation
│   │   ├── persona_gen.py     # stratified sampler + async LLM bio generation
│   │   └── persona_qa.py      # 5-prompt QA gate per persona
│   ├── phase2/
│   │   ├── graph.py           # Watts-Strogatz builder + small-world validation
│   │   ├── influencers.py     # archetype injection + propagation multipliers
│   │   └── schemas.py         # SimGraph, NodeMeta, InfluencerArchetype
│   ├── phase3/
│   │   ├── schemas.py         # ProductStimulus, AgentMemory, PeerSignal, AgentDecision
│   │   ├── memory.py          # MemoryStore + InMemoryBackend + RedisMemoryBackend
│   │   └── feed.py            # personalized per-agent MarketplaceFeed builder
│   ├── phase4/
│   │   ├── loop.py            # run_simulation() + SimulationLog + TimestepLog
│   │   ├── prompts.py         # system + user prompt templates for agent decisions
│   │   ├── decisions.py       # LLM output parser with fuzzy fallback
│   │   └── propagation.py     # social signal fan-out + salience decay
│   ├── phase5/                # validation & calibration [planned]
│   ├── phase6/                # analytics & output [planned]
│   └── gui/
│       └── sim_dashboard.py   # Streamlit dashboard
├── templates/
│   └── persona_bio.j2         # Jinja2 biography template
├── tests/
│   ├── test_phase1/           # 9 tests
│   └── test_phase2/           # 11 tests
├── .env.example
├── pyproject.toml
├── CLAUDE.md                  # architecture reference for AI assistants
└── NEXT_STEPS.md              # full implementation roadmap
```

---

## Research Foundation

This system is grounded in the He et al. (2024) *Societies.io* study (British Journal of Psychology), which demonstrated that 33,299 LLM-powered chatbots on Chirper.ai — given only persona prompts with no explicit social rules — spontaneously reproduced empirically measured homophily, community formation, and information cascade patterns from real social networks.

**Why ABM and not a Genetic Algorithm?** A GA converges toward a homogeneous "super-buyer" population by selecting for fitness. ABM preserves population heterogeneity — agents persist, information propagates through a social structure, and diversity of adoption timing and rejection reasoning is what makes the output analytically useful. Market heterogeneity *is* the signal.

Additional methodology: Toubia et al. (2025) and Arora et al. (2025) demonstrate that prompt-augmented LLMs outperform fine-tuned models for behavioral prediction tasks — LaunchLens uses prompt engineering exclusively, not fine-tuning.

---

## Simulation Parameters

| Parameter | Default | Range | Notes |
|---|---|---|---|
| Agents per run | 1,000 | 100–5,000 | Start with 100 for dev |
| Timestep duration | 1 week | — | Real-world calendar equivalent |
| Run length | 12 timesteps | 4–24 | 3–6 months of market activity |
| Social graph degree k | 6 | 6–10 | Product-relevant social circle |
| Rewiring probability β | 0.15 | 0.1–0.3 | Cross-demographic bridge edges |
| Salience decay per hop | 30% | — | `_SALIENCE_DECAY = 0.70` |
| Salience floor | 0.05 | — | Signals below this are dropped |
| LLM batch size | 200 | 100–500 | Async, semaphore-controlled |
| Target throughput | 50–100 agents/s | — | AWS c6i.2xlarge |
