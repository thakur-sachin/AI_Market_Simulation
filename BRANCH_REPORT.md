# AI_Market_Simulation Branch Report

Audit date: 2026-07-29

Branches inspected:

| Branch | Commit | Relationship | Purpose |
|---|---:|---|---|
| `origin/main` | `ac55509` | Current synced default branch | Baseline LaunchLens prototype plus data requirements plan |
| `origin/st/ML-API-Framework` | `dd88254` | Diverges from `8545cbc init commit` | Adds local/remote LLM provider abstraction, real-data source loaders, calibration, CI, and expanded tests |
| `origin/st/ML-Combined-Framework` | `c237aa0` | 6 commits from `8545cbc init commit`; separate line from `main` | Combines provider/calibration work with Phase 6 analytics/reporting, price A/B flag, cleaner generated-file handling |

Important topology note: `main` and the two `st/*` branches share the original `8545cbc init commit`, but `main` has only one extra commit, `data_requirements`. The feature branches do not contain `main`'s `launchlens_data_requirements_plan.md`; `ML-API-Framework` explicitly deletes it relative to `main`.

## Executive Summary

`main` is the lightest branch. It contains the original simulation architecture for Phases 1-4, a Streamlit dashboard, minimal tests for schemas and graph building, and a large planning document. It presents Phase 5 and Phase 6 mostly as planned work.

`st/ML-API-Framework` is the first real engineering expansion. It wires a provider-agnostic LLM layer with `mock`, `local` Ollama, `sarvam`, and `claude` engines; adds cost estimates, caching, source provenance for district data, Phase 5 calibration/bias modules, CI config, scripts, sample outputs, and significantly broader tests. It also commits generated artifacts like `.coverage`, PDFs, `outputs/`, and `__pycache__`, which should not stay in the final branch.

`st/ML-Combined-Framework` is the best functional base. It keeps most API-framework functionality, removes many generated artifacts, adds `.gitignore`, adds Phase 6 `analytics.py` and `report.py`, improves `run-sim` with `--price-multiplier`, `--calibrate`, `--skip-personas-llm`, model override handling, and report emission. It still has some questionable tracked `.venv` symlinks and removes CI/pre-commit files that the API branch had.

Tests were not executed locally because the system Python has no `pytest` installed. I inspected test files and command surfaces, but no test pass/fail claim is made from this machine.

## Project Concept

The repository implements LaunchLens Labs: a synthetic market intelligence simulator for Indian consumer product launches. The system creates demographically grounded personas, connects them in a social graph, exposes them to a product stimulus and peer signals, asks an LLM or mock engine for each agent's decision at each timestep, propagates social signals, and evaluates adoption patterns.

The conceptual pipeline is:

1. Phase 1: Create district profiles and sample personas.
2. Phase 2: Build a homophilous small-world social graph and inject influencer archetypes.
3. Phase 3: Maintain agent memory and build per-agent marketplace feeds.
4. Phase 4: Run the timestep interaction loop and parse agent decisions.
5. Phase 5: Validate and calibrate results against historical product launch fixtures.
6. Phase 6: Produce analytics/reporting deliverables.

## Branch: `origin/main`

### What This Branch Is

`main` is a baseline prototype and planning branch. It has source code for Phases 1-4 and a Streamlit dashboard, but it lacks the richer provider/cost/calibration/reporting implementation in the feature branches.

Source size excluding bytecode/PDF/output artifacts: about 4,769 lines.

### How It Works

The branch can run a lightweight simulation through `launchlens/sim_lite.py`, generate personas from a district profile, construct a social graph, maintain per-agent memory, build feeds, and run a Phase 4 decision loop. The README describes full LLM routing and validation goals, but some of that is aspirational in this branch.

Main execution paths:

| Command/path | Behavior |
|---|---|
| `python -m launchlens.sim_lite --agents N --timesteps T` | Runs an offline heuristic simulation with no API keys or infrastructure |
| `python -m launchlens.cli ingest-census` | Attempts to build district profiles from local Census/NFHS data |
| `python -m launchlens.cli generate-personas --district ...` | Samples demographic vectors and generates biographies |
| `streamlit run launchlens/gui/sim_dashboard.py` | Opens interactive dashboard backed mostly by the lite/local simulation flow |

### File-by-File Responsibilities

| File | Role |
|---|---|
| `README.md` | Product pitch, architecture explanation, quick start, planned Phase 5/6 scope |
| `CLAUDE.md` | Architecture/reference notes for AI-assisted development |
| `NEXT_STEPS.md` | Implementation roadmap and open work |
| `launchlens_data_requirements_plan.md` | Detailed data requirements plan added by `main`'s local commit |
| `pyproject.toml` | Python package metadata and dependencies |
| `templates/persona_bio.j2` | Persona biography template |
| `launchlens/config.py` | Basic settings via `pydantic-settings` |
| `launchlens/llm.py` | Minimal routing/client stubs for Claude/Sarvam-style LLM calls |
| `launchlens/cli.py` | Basic CLI dispatcher for ingest/persona generation |
| `launchlens/sim_lite.py` | Self-contained mock diffusion simulation and terminal charts |
| `launchlens/gui/sim_dashboard.py` | Streamlit dashboard for overview/network/segments |
| `launchlens/phase1/schemas.py` | Pydantic models for age distribution, district profile, demographic vector, persona |
| `launchlens/phase1/data_pipeline.py` | Census/NFHS-oriented district profile builders |
| `launchlens/phase1/persona_gen.py` | Stratified demographic sampling and biography generation |
| `launchlens/phase1/persona_qa.py` | Persona QA prompts and async validation loop |
| `launchlens/phase2/schemas.py` | Graph and node metadata contracts |
| `launchlens/phase2/graph.py` | Watts-Strogatz graph construction, homophily ordering, graph validation |
| `launchlens/phase2/influencers.py` | Assigns influencer archetypes and propagation multipliers |
| `launchlens/phase3/schemas.py` | Product, peer signal, memory, decision, feed models |
| `launchlens/phase3/memory.py` | In-memory and optional Redis-backed memory store |
| `launchlens/phase3/feed.py` | Builds marketplace feed from product stimulus and peer signals |
| `launchlens/phase4/prompts.py` | Builds LLM decision prompt |
| `launchlens/phase4/decisions.py` | Parses LLM decision text into `AgentDecision` |
| `launchlens/phase4/propagation.py` | Propagates BUY/share/complaint signals to neighbors |
| `launchlens/phase4/loop.py` | Async simulation timestep loop |
| `tests/test_phase1/test_schemas.py` | Phase 1 model/sampling tests |
| `tests/test_phase2/test_graph.py` | Phase 2 graph/influencer tests |

### Strengths

- Clear architecture with sensible phase boundaries.
- Offline `sim_lite` path means the project can be demonstrated without external systems.
- Graph/persona primitives are already separated enough for future extension.

### Weaknesses

- Phase 5 and Phase 6 are mostly documentation/planning.
- LLM provider layer is much thinner than the feature branches.
- Tests cover only early phases.
- The branch tracks `__pycache__` files, which should be ignored.

## Branch: `origin/st/ML-API-Framework`

### What This Branch Is

This branch turns the prototype into a more realistic ML/API framework foundation. It adds local Ollama support, remote provider placeholders, engine selection, cost guardrails, response caching, real-data source loaders, Phase 5 validation/calibration, CI config, and expanded tests.

Source size excluding bytecode/PDF/output artifacts: about 8,059 lines.

### How It Works

The CLI becomes the main operator surface:

| Command | Behavior |
|---|---|
| `fetch-data` | Builds a district profile through source-specific loaders and records provenance |
| `generate-personas` | Samples personas and optionally runs LLM-based QA |
| `run-sim` | Runs dry-run heuristic mode or full Phase 4 LLM-backed simulation |
| `calibrate` | Runs a simulation and compares adoption metrics against a calibration case |

The LLM layer resolves engines in this order for `auto`: local Ollama, Sarvam if configured, Claude if configured, then mock. Explicit `sarvam` or `claude` without matching keys raises `MissingAPIKey`. The branch tracks usage and estimated cost and caches identical completions with `diskcache`.

Phase 1 gains a `phase1/sources/` package:

| Source module | Responsibility |
|---|---|
| `census_pca.py` | Reads Census PCA district demographic fields and language files |
| `nfhs.py` | Reads NFHS district health/wealth indicators and maps wealth quintiles to ISEC |
| `trai.py` | Reads state telecom penetration |
| `nsso_datagovindia.py` | Fetches/caches NSSO expenditure data through `datagovindia` |
| `sources/__init__.py` | Orchestrates source fallback chain and provenance |

Phase 5 adds:

| File | Responsibility |
|---|---|
| `phase5/metrics.py` | Adoption deviation, DTW curve distance, top-segment accuracy, regional Spearman, rejection alignment |
| `phase5/bias.py` | Affluence bias, positivity bias, homogeneity/Gini, language sample checks |
| `phase5/calibration.py` | Loads calibration cases and produces tuning recommendations |
| `data/calibration/*.json` | Placeholder ground-truth cases for Paper Boat, Mamaearth, and boAt |

### File/Artifact Additions vs `main`

Major useful additions:

- `.github/workflows/ci.yml`
- `.pre-commit-config.yaml`
- `KNOWN_GAPS.md`
- `ROADMAP.md`
- `scripts/fetch_data_indore.py`
- `scripts/setup_local_models.sh`
- `launchlens/phase1/sources/*`
- `launchlens/phase5/*`
- tests for Phases 1, 3, 4, and 5

Problematic tracked artifacts:

- `.coverage`
- `Societies.io.pdf`
- `LaunchLens_Labs_Implementation_Plan_v2.md.pdf`
- `outputs/*.json`
- many `__pycache__/*.pyc` files

### Strengths

- Much more production-like LLM abstraction.
- Local-first execution is practical for an 8 GB VRAM machine.
- Better parser behavior: malformed model output returns parse failure instead of silently becoming `IGNORE`.
- Calibration and bias modules create a measurable validation loop.
- CI and pre-commit configuration are useful and should be preserved.

### Weaknesses

- Generated artifacts and bytecode are committed.
- Calibration ground truth is explicitly placeholder, so accuracy claims are not yet real.
- Phase 6 is still deferred.
- `calibrate` re-runs a simulation rather than cleanly scoring a previously persisted full SimulationLog.

## Branch: `origin/st/ML-Combined-Framework`

### What This Branch Is

This is the broadest and most usable branch. It keeps the API framework direction, adds Phase 6 analytics/report generation, improves CLI ergonomics, and adds a price-sensitivity A/B experiment knob.

Source size excluding bytecode/PDF/output artifacts: about 8,703 lines.

### How It Works

The branch supports the most complete end-to-end workflow:

```bash
python -m launchlens.cli fetch-data --district MP001 --name Indore --state "Madhya Pradesh"
python -m launchlens.cli run-sim --district MP001 --agents 30 --timesteps 5 --engine local --skip-personas-llm
python -m launchlens.cli run-sim --district MP001 --agents 30 --timesteps 5 --engine local --calibrate paper_boat_aam_panna
python -m launchlens.cli run-sim --district MP001 --agents 30 --timesteps 5 --engine local --price-multiplier 1.5
python -m launchlens.cli calibrate --product paper_boat_aam_panna --sim-log outputs/sim_MP001_30a_5t.json
```

Key runtime behavior:

- `run-sim` can fall back to a hardcoded Indore profile if `MP001` has not been materialized.
- `--skip-personas-llm` avoids LLM biography generation and uses deterministic biography stubs.
- `--price-multiplier` multiplies launch price and MRP while keeping competitor context constant for A/B tests.
- `--calibrate` loads a product fixture, runs validation, and writes a calibration JSON.
- Every simulation writes a full JSON log with timestep decisions, errors, adoption curve, prices, and usage.
- Every simulation writes a Phase 6 markdown report.

### File-by-File Responsibilities

In addition to the API branch responsibilities:

| File | Role |
|---|---|
| `.gitignore` | Ignores Python cache, env files, outputs, coverage, build artifacts |
| `.env.example` | Expanded configuration sample for engines/models |
| `scripts/fetch_real_data.py` | More explicit real-data Indore profile builder |
| `launchlens/phase6/analytics.py` | Objection map, feature importance, message resonance, segment breakdown |
| `launchlens/phase6/report.py` | Generates markdown report with market fit, adoption curve, segments, resonance, feature priority, objections, and validation |
| `launchlens/phase5/__init__.py` | Exposes Phase 5 public API |
| `launchlens/phase5/calibration.py` | Adds `build_sim_summary()` and `calibrate_from_sim_log()` for direct SimulationLog scoring |
| `launchlens/cli.py` | Adds engine/model override cache reload, `--strict`, `--skip-personas-llm`, `--price-multiplier`, `--calibrate`, full output payloads, and report writing |
| `launchlens/llm.py` | Adds `model_override`, `effective_max_concurrent()`, settings reload compatibility |
| `launchlens/phase4/loop.py` | Adds total LLM error tracking, model override propagation, and full decision log support |
| `launchlens/phase4/prompts.py` | Tightens price-skepticism and archetype-specific prior |
| `launchlens/phase4/propagation.py` | Improves decay/idempotency semantics and same-source replacement |

### Differences from `ML-API-Framework`

Useful changes:

- Adds Phase 6 analytics and markdown reporting.
- Adds full per-timestep simulation output instead of summary-only output.
- Adds direct calibration from an existing simulation log.
- Adds price-multiplier experiment support.
- Adds a hardcoded Indore fallback for easier local runs.
- Adds `.gitignore` and removes many generated artifacts/PDFs/sample outputs.
- Improves engine/model override handling by reloading settings.

Regressions or concerns:

- Removes `.github/workflows/ci.yml` and `.pre-commit-config.yaml`, which should probably be restored.
- Tracks `.venv/bin/python`, `.venv/bin/python3`, `.venv/bin/python3.12`, `.venv/lib64`, and `.venv/pyvenv.cfg`; virtual environments should not be committed.
- README says 96 tests in the badge but the test table still lists 106 tests, inherited from the API branch. This documentation is inconsistent.
- Some UX strings use Unicode symbols; that is fine for a CLI, but the repo style should be consistent.

## Phase-by-Phase Comparison

| Phase | `main` | `ML-API-Framework` | `ML-Combined-Framework` |
|---|---|---|---|
| Phase 1 data | Legacy profile builder only | Adds source chain with Census/NFHS/TRAI/NSSO provenance | Same source chain plus stricter CLI and fallback improvements |
| Phase 1 personas | Stratified vectors and LLM biographies | Adds hard diversity gate and provider-aware generation | Adds skip-LLM persona option for practical local runs |
| Phase 2 graph | Watts-Strogatz + influencers | Mostly same, small parameter/doc adjustments | Mostly same |
| Phase 3 memory/feed | In-memory/Redis memory and feed building | Adds tests and small feed/memory fixes | Adds schema/feed refinements and stronger tests |
| Phase 4 loop | Async LLM decisions with parser/propagation | Provider-aware loop, parse failure accounting, engine concurrency | Adds model override, LLM error counts, full decision output |
| LLM layer | Thin route/client abstraction | Full provider abstraction, cache, cost tracking, mock/local/Sarvam/Claude | Same idea with model override and concurrency helper |
| Phase 5 | Placeholder package | Metrics, bias suite, calibration cases | Adds SimulationLog adapter and integrated calibration flow |
| Phase 6 | Placeholder package | Deferred | Implements analytics primitives and markdown report generation |
| Dashboard | Streamlit overview/network/segments | Provider-aware dashboard, larger feature set | Reworked dashboard aligned with combined branch flow |
| Tests | Phase 1/2 only | Broad tests across 1/3/4/5 | Similar tests, adjusted for new prompt/propagation behavior |

## Recommended Branch Strategy

Use `origin/st/ML-Combined-Framework` as the functional base for future work, but clean it before merging:

1. Restore CI/pre-commit from `ML-API-Framework`.
2. Remove tracked `.venv` entries.
3. Ensure `.gitignore` covers `.venv/`, `__pycache__/`, `*.pyc`, `.coverage`, `outputs/`, and large generated PDFs.
4. Decide whether to bring `launchlens_data_requirements_plan.md` from `main` into the combined branch as documentation.
5. Fix README test-count inconsistency.
6. Install dependencies in a virtual environment and run:

```bash
python -m pytest
ruff check .
ruff format --check .
```

## Merge Risk

The branches are structurally divergent but conceptually compatible. The safest path is not to merge all three blindly. Instead:

- Start from `st/ML-Combined-Framework`.
- Cherry-pick or manually copy the useful docs from `main`, especially `launchlens_data_requirements_plan.md`.
- Restore CI/pre-commit files from `st/ML-API-Framework`.
- Remove generated artifacts before opening a PR.

## Local Verification Performed

Commands run:

- `git fetch origin`
- `git worktree add /private/tmp/ai_market_branch_main origin/main`
- `git worktree add /private/tmp/ai_market_branch_api origin/st/ML-API-Framework`
- `git worktree add /private/tmp/ai_market_branch_combined origin/st/ML-Combined-Framework`
- `rg --files` inventories for all branches
- `git diff --stat` and `git diff --name-status` branch comparisons
- Direct reads of README, pyproject, CLI, LLM, Phase 4, Phase 5, Phase 6, and test surfaces

Test execution attempted:

```bash
python3 -m pytest
```

Result:

```text
/Library/Developer/CommandLineTools/usr/bin/python3: No module named pytest
```

So tests were not executed in this environment.
