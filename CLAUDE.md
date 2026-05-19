# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LaunchLens Labs — Synthetic Market Intelligence Engine. Simulates district-level Indian consumer behavior using an Agent-Based Model (ABM) powered by LLM personas. Long-term target: <8% deviation from real-world adoption outcomes on Phase 5 calibration cases (machinery implemented; real ground truth still to be collected).

Methodological foundation: He et al. (2024) *Societies.io* (British Journal of Psychology) — LLM agents reproduce homophily and community formation from persona prompts alone, without explicit social rules.

**Why ABM not GA:** A genetic algorithm converges to a homogeneous "super-buyer" population. ABM preserves heterogeneity — agents persist, information propagates (not fitness). Market simulation depends on that diversity.

**Hardware reality:** Development happens on an **8 GB VRAM NVIDIA laptop**, no API keys configured. All LLM functionality runs locally via **Ollama** (OpenAI-compatible endpoint at `http://localhost:11434/v1`). Remote providers (Sarvam, Claude) are placeholders that activate automatically when their API keys are set. See `project_hardware.md` memory and `ROADMAP.md` for the scale-up path.

---

## Architecture (5 phases active, Phase 6 deferred)

```
Census / NFHS / TRAI / NSSO data       (real CSVs go in data/raw/)
        │
        ▼
[Phase 1] Diversity Engine
  - phase1/sources/{census_pca, nfhs, trai, nsso_datagovindia}.py
  - load_district_profile_chain() records per-field provenance:
    {"population": "census", "isec_distribution": "nfhs", ...}
  - Stratified persona sampling → Jinja2 biographies via LLM
  - enforce_population_diversity() raises on >5% marginal drift
        │
        ▼
[Phase 2] Social Graph
  - Watts-Strogatz (k=6, β=0.15), homophily-ordered ring
  - inject_influencers(): family_elder / shopkeeper / micro_influencer / whatsapp_hub
  - NetworkX in-memory (Neo4j path deferred — see ROADMAP.md)
        │
        ▼
[Phase 3] Sim Environment
  - ProductStimulus → per-agent MarketplaceFeed
  - MemoryStore: in-memory dict (default) OR Redis (when REDIS_URL set)
  - Tier-2 semantic memory (pgvector) NOT implemented; see ROADMAP.md
        │
        ▼
[Phase 4] Interaction Loop
  - Provider-agnostic LLM dispatch (Ollama / Sarvam / Claude / Mock)
  - JSON-mode decision prompt with anti-positivity prior
  - Two-stage parser (JSON first, fielded-text fallback); no silent IGNORE coercion
  - propagate_decisions(): 30% salience decay, 1.5× COMPLAIN boost
  - Hand-rolled asyncio.gather batch loop (LangGraph deferred)
        │
        ▼
[Phase 5] Validation & Calibration
  - phase5/metrics.py: adoption rate, DTW, top-segment, Spearman, rejection alignment
  - phase5/bias.py: affluence, positivity, homogeneity (Gini), language sampling
  - phase5/calibration.py: CalibrationCase loader → CalibrationReport with tuning signals
  - Three shipped placeholder cases under data/calibration/
        │
        ▼
[Phase 6] Output & Analytics  — DEFERRED (ROADMAP.md)
```

---

## What runs on this hardware right now

| Engine | Activation | Cost | Notes |
|---|---|---|---|
| `mock` | always | $0 | Stochastic heuristic via `sim_lite._mock_decision` |
| `local` | Ollama reachable + model pulled | $0 | Default on dev laptop. `qwen2.5:3b-instruct-q4_K_M`, `gemma2:2b`, `sarvam-1:2b` |
| `sarvam` | `SARVAM_API_KEY` set | per-token | Raises `MissingAPIKey` if key absent |
| `claude` | `ANTHROPIC_API_KEY` set | per-token | Raises `MissingAPIKey` if key absent |

`--engine auto` picks the first available in order: local → sarvam → claude → mock. CLI cost preflight requires `--confirm-cost` for any remote run above `cost_confirm_threshold_usd` (default $10).

---

## Validation Loop

Implemented in `launchlens/phase5/`:

| Metric | Implementation | Gate |
|---|---|---|
| Adoption rate deviation | `metrics.adoption_rate_deviation` | < 0.08 |
| Adoption curve shape | `metrics.dtw_curve_distance` (dtaidistance + DP fallback) | < 0.15 |
| Top segment accuracy | `metrics.top_segment_accuracy` | ≥ 2 of 3 |
| Regional Spearman | `metrics.regional_spearman` (scipy + NumPy fallback) | > 0.70 |
| Rejection reason alignment | `metrics.rejection_reason_alignment` (sentence-transformers + Jaccard fallback) | ≥ 2 of 3 |

Bias suite (`phase5/bias.py`):
- `affluence_bias`: flag if lower-tier BUY rate exceeds upper-tier BUY rate beyond threshold
- `positivity_bias`: compare sim REJECT rate to category benchmark
- `homogeneity_gini`: mean within-ISEC-cohort Gini; flag if < 0.30
- `language_bias_sample`: writes JSONL of Indic-language reasoning for human review

`tune_signal()` emits structured recommendations matching this table when a gate fails:

| Failure | Recommendation |
|---|---|
| Adoption too high | Increase price sensitivity weight; reduce `_ISEC_BASE_BUY` for lower tiers; sharpen anti-positivity prior |
| Adoption too low | Increase social proof multiplier; raise salience floor; extend COMPLAIN boost symmetry to BUY |
| Wrong segments | Audit `sample_demographic_vectors`; verify district provenance is not "fallback" |
| Curve too fast | Reduce `_ARCHETYPE_SPEED`; raise advancement threshold |
| Regional misalignment | Audit `_disaggregate_isec`; verify NFHS-5 wealth mapping; check cross-district edges |
| Rejection misalignment | Enrich ProductStimulus context; review `internal_reasoning` samples |

`KNOWN_GAPS.md` lists every uncited constant the system currently relies on.

---

## Key Data Contracts

**`DistrictProfile`** (`phase1/schemas.py`)
- All numeric fields validated 0-1 / sums-to-1 via Pydantic field validators
- `provenance: dict[str, Literal["census", "nfhs", "trai", "nsso", "baseline", "manual", "fallback"]]`
- Built by `phase1/sources/load_district_profile_chain(district_id, name, state)`

**`AgentMemory`** (`phase3/schemas.py`)
- `biography: str` — immutable persona text
- `episodic_buffer: list[str]` — rolling 10 entries
- `product_opinion: dict[str, str]`, `purchase_history: list[dict]`, `current_decision: dict[str, DecisionState]`
- `peer_signals: list[PeerSignal]` — propagation inbox
- No `opinion_embedding` (deferred; pgvector not wired)

**`AgentDecision`** — emitted by every LLM call, structured by parser
- 9 states: `IGNORE | AWARE | RESEARCH | CONSIDER | BUY | REJECT | SHARE_POSITIVE | SHARE_NEGATIVE | COMPLAIN`
- Parser returns `None` on failure; caller increments `parse_failures` (never silently coerced)

---

## Tech Stack (current)

| Layer | Technology |
|---|---|
| Data sources | `phase1/sources/{census_pca,nfhs,trai,nsso_datagovindia}.py`; pandas + datagovindia |
| Persona templates | Jinja2 (`templates/persona_bio.j2`) |
| Graph | NetworkX in-memory |
| Memory | In-memory dict (Redis backend optional via `REDIS_URL`) |
| LLM dispatch | `launchlens/llm.py` provider protocol: Ollama / Sarvam / Claude / Mock |
| Response cache | `diskcache` keyed by sha256 of provider+model+prompt |
| NLP analytics | sentence-transformers (best-effort in `phase5/metrics`); BERTopic declared but unused |
| Dashboard | Streamlit (`launchlens/gui/sim_dashboard.py`) with engine selector |
| Validation | scipy + numpy + dtaidistance |
| Testing | pytest, pytest-asyncio, pytest-cov |
| Lint/type | ruff, mypy (advisory) |

Deferred: LangGraph orchestration, pgvector semantic memory, Neo4j, FastAPI, React/Mapbox. See `ROADMAP.md`.

---

## Simulation parameters (8 GB VRAM defaults)

- Agents per run: 50–300 local (config: `default_agent_count=100`)
- Timestep: ~1 real week
- Run length: 4–16 timesteps for local sims (12–24 cloud-scale)
- Graph: k=6, β=0.15
- LLM concurrency: 4 for local, 8 for remote (`llm_max_concurrent_local` / `_remote`)
- Max prompt tokens: 3500 (Q4 3B model + 4K context headroom)
- Salience decay: 0.70 per timestep, floor 0.05, COMPLAIN boost 1.5×

---

## Common commands

```bash
# Install + setup
pip install -e ".[dev]"
bash scripts/setup_local_models.sh

# Data
python scripts/fetch_data_indore.py
python -m launchlens.cli fetch-data --district MP001 --state "Madhya Pradesh"

# Sims
python -m launchlens.cli run-sim --district MP001 --agents 50 --timesteps 4 --dry-run
python -m launchlens.cli run-sim --district MP001 --agents 50 --timesteps 4 --engine local
python -m launchlens.cli calibrate --product paper_boat_aam_panna --district MP001

# Tests + lint
pytest                                      # 106 tests
pytest --cov=launchlens --cov-fail-under=30
ruff check . && ruff format --check .
mypy launchlens/

# UI
streamlit run launchlens/gui/sim_dashboard.py
```

---

## Known Risks (residual)

- **Positivity bias** — `phase4/prompts.py` now includes an explicit anti-positivity prior. Empirical validation pending real-product calibration.
- **Homogeneity** — `enforce_population_diversity` exists; not auto-invoked by CLI. Run with `validate_population_diversity` first.
- **Ungrounded constants** — see `KNOWN_GAPS.md` for the full ledger (ISEC base buy rates, archetype multipliers, propagation decay).
- **Synthetic data** — `data/raw/` is empty by default. Every district profile produced today has 11/11 fallback fields. Drop real CSVs in per `ROADMAP.md` to unlock real provenance.
- **Calibration ground truth** — `data/calibration/*.json` carry placeholder adoption curves. Gates fire correctly but cannot certify accuracy until real curves replace placeholders.
