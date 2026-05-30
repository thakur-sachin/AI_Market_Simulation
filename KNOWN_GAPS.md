# Known Technical Gaps

A living ledger of constants, heuristics, and data assumptions that are not yet grounded. Every numeric value in `launchlens/` should either cite a source in a comment OR appear here.

When a row is moved to a cited value in code, delete it from this list.

---

## Ungrounded numeric constants

### `launchlens/phase1/persona_gen.py`

| Constant | Value | Notes |
|---|---|---|
| `_ISEC_INCOME_RANGE` | 12 (lo, hi) INR pairs | Not derived from NSSO CES deciles or actual MRSI income bands. Plausible-looking but invented. |
| `±15%` stochastic income variation | line ~182 | Arbitrary jitter range. |
| `_ISEC_EDUCATION` mapping | 12 string labels | Approximate; not validated against Census C-08 education tables. |
| `_ADOPTION_ARCHETYPE_MAP` | ISEC → archetype list | Heuristic, not from Rogers-style empirical adoption curves for India. |

### `launchlens/phase1/data_pipeline.py`

| Constant | Value | Notes |
|---|---|---|
| `_ISEC_NATIONAL_BASELINE` | 12 floats | Cited as MRSI 2024 but no source URL or document version. |
| TRAI urban/rural penetration | `0.78 / 0.38` | Hardcoded; should be loaded from `data/raw/trai/trai_state_quarterly.csv`. Already overridable by the new `phase1/sources/trai.py` chain when the file exists. |
| Median expenditure formula | `upper_share * 25000 + (1-upper_share) * 8000` | Linear interpolation with magic numbers; no NSSO grounding. The new NSSO source path supersedes this when `DATAGOVINDIA_API_KEY` is set. |

### `launchlens/phase2/influencers.py`

| Archetype | Awareness × | Trust × | Share | Degree |
|---|---|---|---|---|
| family_elder | 1.0 | 2.0 | 10% | 3–5 |
| local_shopkeeper | 1.5 | 1.0 | 3% | 15–25 |
| micro_influencer | 1.3 | 0.8 | 0.8% | 50–200 |
| whatsapp_hub | 1.2 | 1.0 | 6.5% | 10–15 |

All eight values are uncited. Calibration against real Indian influencer studies (or A/B simulation against historical product spread patterns) is required before claiming predictive validity for runs that depend on propagation dynamics.

### `launchlens/phase4/propagation.py`

| Constant | Value | Notes |
|---|---|---|
| `_SALIENCE_DECAY` | 0.70 | "30% per hop" assumption — invented. |
| `_SALIENCE_FLOOR` | 0.05 | Cutoff threshold — invented. |
| `_COMPLAIN_BOOST` | 1.5 | Negative-information amplification — invented. |

### `launchlens/sim_lite.py` (heuristic engine)

Every constant in this file is uncalibrated. The module is used as the `--dry-run` engine and the dashboard's `mock` engine. Annotated in code with `# UNCALIBRATED` references. Affected constants:

- `_ISEC_BASE_BUY` — 12 base BUY probabilities by tier
- `_ARCHETYPE_SPEED` — 5 multipliers for adoption speed by archetype
- `affordability = income / (price * 6.67)` — magic constant
- `social_boost = pos * 0.08 - neg * 0.05` — invented weights
- Funnel advance probability `p_buy * (0.3 + 0.2 * funnel_idx)` — invented

This engine is acceptable for:
- Shape-validating Phase 4 plumbing (cascade fires when first BUY occurs)
- Dashboard exploration
- CI integration tests

It is **not** acceptable for:
- Calibration metric comparisons against real products
- Any claim about real-world adoption accuracy

---

## Real data — grounded so far

**Indore (MP001)** is grounded from authoritative public sources via `scripts/fetch_real_data.py`:

| Field | Source | Value |
|---|---|---|
| population | Census 2011 | 3,276,697 |
| urban_share | Census 2011 | 74.09% |
| sex_ratio | Census 2011 | 928 |
| literacy_rate | Census 2011 | 80.87% |
| language_distribution | Census 2011 C-16 | Hindi 71.4%, Malvi 15.1%, Marathi 3.5%, Urdu 2.8%, Sindhi 1.7%, Nimadi 1.4%, Gujarati 1.0%, other 3.1% |
| median_monthly_hh_expenditure | HCES 2022-23 Stmt 8 (MP × 1.25 wealth-rank adjustment) | ₹26,446 |
| smartphone_penetration | TRAI Q1 2025 (urban 75% / rural 38% weighted) | 65% |
| internet_penetration | TRAI Q1 2025 | 61% |

Still fallback (need direct district data):
- `isec_distribution` — inferred from NFHS-5 wealth-quintile ranking (Indore is in top-5 richest MP districts) but not directly translated from per-district wealth quintile counts. Requires `data/raw/nfhs/nfhs5_district.csv` to upgrade.
- `age_distribution` — using all-India 2011 pyramid as proxy. District-level age bands not in the public mirrors I could reach; need direct Census C-13 download.
- `upi_adoption` — derived as 65% × internet penetration; NPCI publishes only state aggregates.

Other districts: 11/11 fallback. `scripts/fetch_real_data.py` accepts a `--district` flag and currently only knows MP001.

## Missing calibration ground truth

`data/calibration/*.json` carry placeholder `real_adoption_curve` values. The Phase 5 metric machinery runs against them and is unit-tested, but `--engine local | sarvam | claude` calibration runs cannot honestly report gate pass/fail until real curves replace the placeholders.

## Missing calibration ground truth

`data/calibration/*.json` carry placeholder `real_adoption_curve` values. The Phase 5 metric machinery runs against them and is unit-tested, but `--engine local | sarvam | claude` calibration runs cannot honestly report gate pass/fail until real curves replace the placeholders.

---

## Deferred validation

| Concern | Where | Status |
|---|---|---|
| Diversity gate (<5% drift) | `phase1/persona_gen.enforce_population_diversity` | Function implemented; not auto-invoked. CLI `generate-personas` still uses the warn-only `validate_population_diversity`. |
| Small-world σ precondition | `phase2/graph.validate_small_world:182` | `path_rand = log(n)/log(k)` formula is valid only for sparse graphs (`k < log(n)`); no assertion added yet. |
| QA failure auto-regeneration | `phase1/persona_qa` | Currently filters failed personas out; does not regenerate. |
| pgvector semantic memory | `phase3/memory` | Tier-1 only; semantic retrieval not implemented. |
| LangGraph orchestration | `phase4/loop` | Hand-rolled async loop instead. |
| Cross-validation framework | Phase 5 | Single test/calibration set; no held-out test split. |

---

## Open methodological questions

1. **Salience decay timing** — `propagate_decisions` applies one decay step over *all* signals (newly written + carried over). Should newly emitted signals start at full salience and only decay on the *next* timestep? Affects how quickly cascades cool.
2. **Anti-positivity prior strength** — the JSON-mode prompt now instructs default-to-RESEARCH/IGNORE. Empirically validate against real adoption rates once calibration data is collected; the prior may need to be moderated for premium A-tier consumers.
3. **NFHS wealth-quintile → ISEC mapping** — `nfhs.wealth_quintiles_to_isec` uses a fixed split (e.g. Q5 → A1×0.15, A2×0.25, A3×0.35, B1×0.25). The split fractions are heuristic; NSSO CES decile data should be used as a check.
4. **UPI adoption derivation** — `phase1/sources/__init__:upi_adoption = 0.60 * urban_internet + 0.35 * rural_internet`. The 0.60 / 0.35 weights are guesses; NPCI publishes only state-level monthly totals.
