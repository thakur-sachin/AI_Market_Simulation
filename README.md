# LaunchLens Labs

**Synthetic Market Intelligence Engine for Indian Consumer Markets**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Phase](https://img.shields.io/badge/phases-1--5%20wired-brightgreen.svg)](#project-status)
[![Tests](https://img.shields.io/badge/tests-96%20passing-brightgreen.svg)](#running-tests)
[![Methodology](https://img.shields.io/badge/methodology-He%20et%20al.%202024-purple.svg)](#research-foundation)

---

LaunchLens simulates how Indian consumers discover, evaluate, and decide on new products. It builds a synthetic population of LLM-powered agents drawn from Census, NFHS, and TRAI data, connects them in a Watts-Strogatz social network, and runs a week-by-week diffusion simulation. Phase 5 validates the simulation against historical Indian product launches.

**Current hardware target:** local development on an 8 GB VRAM NVIDIA laptop using **Ollama** with Qwen2.5-3B-Instruct (Q4). Remote providers (Sarvam, Claude) are wired as placeholders and activate automatically when their API keys are configured.

**Long-term accuracy target:** <8% deviation from real-world adoption outcomes on Phase 5 calibration cases (currently scaffolded with placeholder ground truth; see `data/calibration/` and `KNOWN_GAPS.md`).

---

## Pipeline

```
Census / NFHS-5 / TRAI / NSSO data        (real CSVs go in data/raw/)
            │
            ▼
┌─────────────────────────────┐
│  Phase 1 · Diversity Engine │  per-field provenance: which source supplied what.
│                             │  Stratified persona sampling. Diversity gate <5% drift.
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 2 · Social Graph     │  Watts-Strogatz small-world (k=6, β=0.15).
│                             │  Homophily ordering + 4 influencer archetypes.
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 3 · Sim Environment  │  ProductStimulus → per-agent feed.
│                             │  In-memory MemoryStore (Redis optional).
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 4 · Interaction Loop │  LLM decision per agent (Ollama / Sarvam / Claude / mock).
│                             │  Anti-positivity prior · JSON-mode parser · 30% salience decay.
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 5 · Validation       │  5-metric calibration (adoption / DTW / segments /
│                             │  Spearman / rejection alignment) + bias suite.
└─────────────────────────────┘
```

Phase 6 (NLP analytics, FastAPI, React dashboard) is **deferred** — see [`ROADMAP.md`](ROADMAP.md).

---

## Quick Start (8 GB VRAM laptop, no API keys)

```bash
git clone <repo>
cd LaunchLensLab
pip install -e ".[dev]"

# 1. Pull local models (~4 GB total). Requires Ollama running.
bash scripts/setup_local_models.sh

# 2. Build a district profile for Indore — uses real data if you've placed CSVs,
#    otherwise marks every field 'fallback' so you can see what's synthetic.
python scripts/fetch_data_indore.py

# 3. Dry-run sim (heuristic decisions, no LLM). Works even if Ollama is stopped.
python -m launchlens.cli run-sim --district MP001 --agents 50 --timesteps 4 --dry-run

# 4. Local-LLM sim. With --calibrate, also scores against a real-launch fixture
#    and emits a markdown report.
python -m launchlens.cli run-sim --district MP001 --agents 50 --timesteps 4 \
    --engine local --calibrate paper_boat_aam_panna

# 5. Re-score an already-saved sim log against a different calibration case.
python -m launchlens.cli calibrate --product mamaearth_vitc \
    --sim-log outputs/sim_MP001_50a_4t.json

# 6. Dashboard
streamlit run launchlens/gui/sim_dashboard.py
```

Run outputs land in `./outputs/`:
- `sim_<district>_<n>a_<t>t.json` — full SimulationLog (per-timestep decisions + usage tracker)
- `calibration_<product>.json` — 5-gate validation + tuning recommendations
- `report_<district>_<n>a_<t>t.md` — 8-deliverable markdown report

### Engine selection

`--engine {auto, mock, local, sarvam, claude}` (default `auto`). Auto resolves in order:
1. Local Ollama if a matching model is pulled.
2. Sarvam if `SARVAM_API_KEY` is set.
3. Claude if `ANTHROPIC_API_KEY` is set.
4. Mock (deterministic JSON IGNORE response) as the always-available fallback.

`--engine sarvam` or `--engine claude` without the matching API key raises a clear `MissingAPIKey` error rather than silently degrading.

### Cost guardrails

`estimate_cost()` runs as a preflight before every `run-sim`. Local and mock are always $0. Remote runs above `cost_confirm_threshold_usd` (default $10) require `--confirm-cost`. Every completion is logged in `LLMUsageTracker` and surfaced as a KPI in the dashboard.

Disk-backed `diskcache` keys responses by `sha256(provider + model + system + user + temp + json_mode)` — identical re-runs cost nothing.

---

## Architecture Reference

### Phase 1 · Diversity Engine

| Module | Purpose |
|---|---|
| `phase1/schemas.py` | `DistrictProfile` with `provenance: dict[field, source]`; Pydantic validators on sums/bounds |
| `phase1/data_pipeline.py` | Legacy single-source builder; kept for backward compatibility |
| `phase1/sources/` | Per-source loaders (Census PCA, NFHS-5, TRAI, NSSO via `datagovindia`) |
| `phase1/sources/__init__.py` | `load_district_profile_chain()` orchestrator; per-field provenance; `strict=True` raises on any fallback |
| `phase1/persona_gen.py` | Stratified sampler; `enforce_population_diversity` (hard gate, <5% drift) |
| `phase1/persona_qa.py` | 5-prompt QA per persona, wired through `--qa` on CLI |

**Data files (not shipped):** drop into `data/raw/{census,nfhs,trai,nsso}/`. Source URLs are printed by `scripts/fetch_data_indore.py` when a file is missing.

### Phase 2 · Social Graph

| Module | Purpose |
|---|---|
| `phase2/graph.py` | Watts-Strogatz builder with homophily pre-ordering; small-world σ validation |
| `phase2/influencers.py` | Injects 4 influencer archetypes; rewires edges to target degree ranges |

**Influencer multipliers (UNCALIBRATED — see `KNOWN_GAPS.md`):**

| Archetype | Share | Degree | Awareness mult | Trust mult |
|---|---|---|---|---|
| Family Elder | 10% | 3–5 | 1.0× | 2.0× |
| Local Shopkeeper | 3% | 15–25 | 1.5× | 1.0× |
| Micro-Influencer | 0.8% | 50–200 | 1.3× | 0.8× |
| WhatsApp Hub | 6.5% | 10–15 | 1.2× | 1.0× |

### Phase 3 · Simulation Environment

| Module | Purpose |
|---|---|
| `phase3/schemas.py` | `ProductStimulus`, `AgentMemory`, `PeerSignal`, `AgentDecision`, `MarketplaceFeed` |
| `phase3/memory.py` | `MemoryStore` (in-memory by default; optional Redis backend behind `REDIS_URL`) |
| `phase3/feed.py` | Per-agent feed: 1 ad + ≤5 peer reviews + ≤3 purchases + competitor + noise; deduplicated by source |

> **Note:** Two-tier (Redis + pgvector) memory is not implemented. Only Tier-1 episodic exists today. See `ROADMAP.md`.

### Phase 4 · Interaction Loop

| Module | Purpose |
|---|---|
| `phase4/loop.py` | Async batch loop; engine-aware concurrency limits; `SimulationLog.total_parse_failures` |
| `phase4/prompts.py` | JSON-output decision prompt with **anti-positivity prior** |
| `phase4/decisions.py` | Two-stage parser (JSON first, fielded-text fallback); returns `None` on failure (no silent IGNORE coercion) |
| `phase4/propagation.py` | Social fan-out; 30% salience decay per hop; 1.5× COMPLAIN boost |
| `sim_lite.py` | Self-contained stochastic heuristic engine — no LLM required |

**9 decision states**, propagating: `BUY`, `SHARE_POSITIVE`, `SHARE_NEGATIVE`, `COMPLAIN`.

### Phase 5 · Validation & Calibration

| Module | Purpose |
|---|---|
| `phase5/metrics.py` | adoption rate deviation, DTW (via `dtaidistance`, with DP fallback), top-segment accuracy, regional Spearman, rejection alignment (sentence-transformers, with Jaccard fallback) |
| `phase5/bias.py` | Affluence, positivity, homogeneity (Gini), language sample-for-review |
| `phase5/calibration.py` | `CalibrationCase` loader → `CalibrationReport` with gates and structured tuning signals |
| `data/calibration/*.json` | Three shipped cases (Paper Boat / mamaearth / boAt) with **placeholder ground truth** — real curves still to be collected |

**Gates (default):**
| Metric | Gate |
|---|---|
| Adoption rate deviation | < 0.08 |
| DTW curve distance | < 0.15 |
| Top-3 segment accuracy | ≥ 2 of 3 |
| Regional Spearman | > 0.70 |
| Rejection alignment | ≥ 2 of 3 |

---

## LLM Layer

Provider-agnostic dispatch via `launchlens.llm.complete()`:

```python
from launchlens.llm import LLMRoute, complete

text = await complete(
    route=LLMRoute.SARVAM,            # routing hint (English-premium → CLAUDE, else SARVAM)
    system=prompt_system,
    user=prompt_user,
    json_mode=True,
    engine_override="auto",           # auto | mock | local | sarvam | claude
)
```

| Provider | Activation | Cost |
|---|---|---|
| `OllamaProvider` | Reachable Ollama + model pulled | $0 |
| `SarvamProvider` | `SARVAM_API_KEY` set | per-token list price |
| `ClaudeProvider` | `ANTHROPIC_API_KEY` set | per-token list price |
| `MockProvider` | Always (last-resort fallback) | $0 |

Default local models (8 GB VRAM safe): `qwen2.5:3b-instruct-q4_K_M` (multilingual, default route), `gemma2:2b-instruct-q4_K_M` (fast), `sarvam-1:2b` (Indic, if imported via `ollama create`).

---

## Configuration

```bash
cp .env.example .env
```

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Local Ollama OpenAI-compatible endpoint |
| `OLLAMA_DEFAULT_MODEL` | `qwen2.5:3b-instruct-q4_K_M` | Default route |
| `OLLAMA_INDIC_MODEL` | same as default | Override to `sarvam-1:2b` once imported |
| `OLLAMA_FAST_MODEL` | `gemma2:2b-instruct-q4_K_M` | Smaller/faster |
| `LAUNCHLENS_ENGINE` | `auto` | `auto / mock / local / sarvam / claude` |
| `SARVAM_API_KEY` | *unset* | Activates `SarvamProvider` when set |
| `ANTHROPIC_API_KEY` | *unset* | Activates `ClaudeProvider` when set |
| `REDIS_URL` | *unset* | Optional Tier-1 episodic memory backend |
| `LLM_MAX_CONCURRENT_LOCAL` | 4 | Per-VRAM concurrency limit for Ollama |
| `LLM_MAX_CONCURRENT_REMOTE` | 8 | Default concurrency for paid providers |
| `MAX_PROMPT_TOKENS` | 3500 | Q4 3B model + 4K context headroom |

---

## Running Tests

```bash
pytest                            # all 106 tests
pytest tests/test_phase5          # validation metrics only
pytest -v -k "parse"              # by keyword
pytest --cov=launchlens           # with coverage report
```

| Suite | Tests | Status |
|---|---|---|
| `tests/test_phase1/test_schemas.py` | 9 | ✅ |
| `tests/test_phase1/test_sources.py` | 6 — provenance chain, schema validators | ✅ |
| `tests/test_phase2/test_graph.py` | 11 | ✅ |
| `tests/test_phase3/test_memory.py` | 7 | ✅ |
| `tests/test_phase3/test_feed.py` | 6 | ✅ |
| `tests/test_phase4/test_decisions.py` | 13 — JSON & fielded parser, no silent fallback | ✅ |
| `tests/test_phase4/test_prompts.py` | 4 — anti-positivity prior + JSON output | ✅ |
| `tests/test_phase4/test_propagation.py` | 7 — decay, COMPLAIN boost, idempotency | ✅ |
| `tests/test_phase4/test_loop_e2e.py` | 2 — deterministic-LLM cascade + parse failure surfacing | ✅ |
| `tests/test_phase4/test_llm_provider.py` | 9 — provider selection, cost, missing keys | ✅ |
| `tests/test_phase5/test_metrics.py` | 18 — 5 metrics, known-answer fixtures | ✅ |
| `tests/test_phase5/test_bias.py` | 7 — affluence / positivity / homogeneity | ✅ |
| `tests/test_phase5/test_calibration.py` | 7 — case loading, report, tuning signals | ✅ |

CI runs `ruff check`, `ruff format --check`, `mypy launchlens/` (advisory), and `pytest --cov-fail-under=30` on every PR.

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| 1 | Diversity Engine (sources + provenance) | ✅ Implemented; **needs real CSVs** for live profiles |
| 2 | Social Graph | ✅ Implemented; influencer multipliers uncalibrated |
| 3 | Memory + Feed (Tier-1 only) | ✅ Implemented; Tier-2 pgvector deferred |
| 4 | Interaction Loop (mock + Ollama + remote placeholders) | ✅ Implemented |
| 5 | Validation & Calibration | ✅ Machinery implemented; **needs real ground truth** |
| 6 | NLP analytics, FastAPI, React | 🔲 Deferred — see `ROADMAP.md` |

---

## Documentation

- `CLAUDE.md` — architecture reference (auto-loaded by Claude Code sessions)
- `ROADMAP.md` — deferred items: pgvector, LangGraph, Neo4j, FastAPI, React, NLP analytics
- `KNOWN_GAPS.md` — ungrounded constants and missing data, treated as a living ledger
- `NEXT_STEPS.md` — older roadmap (kept for historical context; superseded by `ROADMAP.md`)

---

## Research Foundation

Grounded in He et al. (2024) *Societies.io* (British Journal of Psychology): 33,299 LLM-powered chatbots given only persona prompts spontaneously reproduced empirically measured homophily, community formation, and information cascade patterns.

**Why ABM and not a Genetic Algorithm?** A GA converges toward a homogeneous "super-buyer" by selecting for fitness. ABM preserves population heterogeneity — agents persist, information propagates, and diversity of adoption timing and rejection reasoning is what makes the output analytically useful.

Additional methodology: Toubia et al. (2025), Arora et al. (2025) — prompt-augmented LLMs outperform fine-tuned models for behavioral prediction.
