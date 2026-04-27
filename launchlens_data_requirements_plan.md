# LaunchLens Data Requirements And Acquisition Plan

Inspected codebase: `AI_Market_Simulation.zip`, unpacked at `work/unpacked/AI_Market_Simulation`.

## Executive Diagnosis

The repository can run a lightweight synthetic simulation today, but it does not yet have the data needed to make reliable market predictions.

There are three distinct data layers:

1. **Population data**: Required to build district-level synthetic people. This is partially defined in code and can be filled from public/official sources.
2. **Behavioral decision data**: Required to estimate purchase probability, adoption timing, sharing, rejection, and repeat behavior. This is mostly missing and currently replaced by constants.
3. **Historical launch ground truth**: Required to calibrate and prove accuracy. This is planned but not implemented or bundled.

The code currently supports adoption-rate style outputs from simulated decisions. It does not yet support defensible expected sales, CAC, repeat purchase, market sizing, or multi-year forecasts because the required data contracts and model fields are absent.

## Repository Evidence

### Phase 1: District Profiles And Personas

Relevant files:

- `launchlens/phase1/schemas.py`
- `launchlens/phase1/data_pipeline.py`
- `launchlens/phase1/persona_gen.py`
- `templates/persona_bio.j2`

`DistrictProfile` requires:

| Field | Exact requirement | Source needed | Current status |
|---|---|---|---|
| `district_id` | Census district code | Census PCA district table | Required by loader |
| `district_name` | District name | Census PCA | Required by loader |
| `state_name` | State name | Census PCA | Required by loader |
| `population` | Total population | Census PCA | Required by loader |
| `age_distribution` | `age_0_4`, `age_5_14`, `age_15_24`, `age_25_34`, `age_35_44`, `age_45_54`, `age_55_64`, `age_65_plus` as shares or percentages | Census C-13/C-14 or equivalent age table | Optional in loader; falls back to national baseline |
| `sex_ratio` | Female/male ratio per 1000 males | Census PCA male/female population | Computed |
| `urban_share` | Urban population / total population | Census PCA | Computed |
| `literacy_rate` | Literate population / total population | Census PCA | Computed |
| `language_distribution` | Mother tongue shares by district | Census C-16 | Optional; falls back to Hindi 100% |
| `isec_distribution` | ISEC/NCCS A1-E3 shares | MRSI/NCCS, household assets, education and occupation mapping | Currently estimated from urbanization and literacy |
| `median_monthly_hh_expenditure` | District household monthly expenditure | NSS CES, household consumption survey, CMIE/ICE 360, or primary survey | Currently rough formula |
| `smartphone_penetration` | District smartphone/mobile internet ownership | NFHS-5, TRAI, IAMAI/Kantar, primary survey | Partially estimated |
| `internet_penetration` | District internet penetration | TRAI/IAMAI/NFHS | Currently `smartphone * 0.80` |
| `upi_adoption` | UPI use by district/consumer type | NPCI, RBI, fintech/payment partner, primary survey | Currently `internet * 0.60` |

Minimum files the current loader expects:

- `data/raw/census/district_pca.csv`
- `data/raw/census/c16_language.csv`
- `data/raw/nfhs/nfhs5_district.csv`

Exact required `district_pca.csv` columns:

```text
district_code
district_name
state_name
total_population
male_population
female_population
urban_population
literate_population
```

Optional age columns if you do not want national fallback:

```text
age_0_4
age_5_14
age_15_24
age_25_34
age_35_44
age_45_54
age_55_64
age_65_plus
```

Exact `nfhs5_district.csv` column currently used:

```text
district_code
mobile_internet_women
```

Additional NFHS fields you should add even though the code does not yet use them:

```text
women_mobile_phone_ownership
internet_ever_used_women
households_electricity
households_improved_sanitation
households_clean_fuel
health_insurance
women_literacy
men_literacy
```

These can improve wealth, media access, and category-affinity proxies.

### Current Phase 1 Placeholder Assumptions

The following are not real district data:

| Code behavior | Why it matters |
|---|---|
| ISEC distribution is shifted from a national baseline using only urban share and literacy | This will mis-rank affluent but less urban districts, high-income industrial districts, and migrant-heavy metros |
| Missing age data falls back to national age baseline | City age structure is central for skincare, beverages, electronics, and D2C categories |
| Missing language data falls back to Hindi 100% | Breaks Bengaluru, Chennai, Hyderabad, Kolkata, Kerala, Northeast, and mixed-metro realism |
| Smartphone penetration uses fixed urban/rural constants | Too coarse for city/district comparison |
| Internet and UPI are simple multipliers | Not defensible for purchase-channel or CAC prediction |
| Median household expenditure is derived from ISEC share using two constants | Not a consumption model |
| Occupation is sampled from simple age/urban/ISEC rules | Needs worker census, PLFS, NSS, or survey calibration |
| Tech adoption archetype is assigned only by ISEC | Needs actual category trial/digital behavior data |

### Phase 2: Social Graph And Influence

Relevant files:

- `launchlens/phase2/graph.py`
- `launchlens/phase2/influencers.py`
- `launchlens/phase2/schemas.py`

Current model:

- Agents are sorted by district, ISEC band, age band, and language before a Watts-Strogatz graph is built.
- Default graph degree is typically `k=6` or `k=8`.
- Rewiring probability is usually `beta=0.15`.
- Cross-district edges can be added at 1-3% or 2-5% of edges.
- Influencer archetypes are hard-coded:

| Archetype | Current share | Degree range | Awareness multiplier | Trust multiplier |
|---|---:|---:|---:|---:|
| `family_elder` | 10.0% | 3-5 | 1.0 | 2.0 |
| `local_shopkeeper` | 3.0% | 15-25 | 1.5 | 1.0 |
| `micro_influencer` | 0.8% | 50-200 | 1.3 | 0.8 |
| `whatsapp_hub` | 6.5% | 10-15 | 1.2 | 1.0 |

Data required to calibrate this:

| Parameter | Dataset needed |
|---|---|
| Average degree by segment | Survey: “who influences your purchase decisions?”, household/social circle size, WhatsApp group behavior |
| Homophily by age, income, language, geography | Primary survey or social/contact diary study |
| Influencer prevalence | Survey plus social media/influencer market benchmarks |
| Trust multipliers | Conjoint/discrete choice tests with source of recommendation varied |
| Cross-district edges | Commute, migration, transport links, college/work networks, e-commerce sharing behavior |
| Negative-word-of-mouth salience | Review/referral experiments and complaint sharing data |

Pragmatic MVP: run a 1,500-respondent influence survey across the first 5 cities and ask respondents to rank trust in family, friends, shopkeeper, influencer, WhatsApp group, retailer review, marketplace rating, and doctor/expert depending on category.

### Phase 3: Product Stimulus And Marketplace Feed

Relevant files:

- `launchlens/phase3/schemas.py`
- `launchlens/phase3/feed.py`

Current `ProductStimulus` fields:

```text
product_id
product_name
category
price_mrp
price_launch
currency
key_features
distribution_channels
marketing_copy
competitor_context
target_segment
```

This is enough for an LLM-style qualitative simulation. It is not enough for expected sales, CAC, repeat purchase, distribution performance, or long-term forecasting.

Add these data contracts:

| New schema | Fields needed | Why |
|---|---|---|
| `CategoryMarketProfile` | category, city/district, penetration, monthly/quarterly purchase frequency, average spend, premium share, online/offline split, seasonality index | Converts generic product interest into category-specific demand |
| `PriceBenchmark` | category, pack size, price, brand, channel, rating, review count, discount depth, bestseller rank | Calibrates price sensitivity and competitor context |
| `DistributionPlan` | channel, city/district availability, store count, pin-code coverage, delivery SLA, stock availability, retail margin | Needed before expected sales can be computed |
| `MediaPlan` | channel, spend, CPM/CPC/CPV, reach, frequency, targeting, city split, creative variant | Needed for awareness, CAC, and adoption timing |
| `ChannelConversionBenchmark` | channel, category, city, segment, CTR, CVR, add-to-cart, checkout, CAC | Needed for acquisition model |
| `ConceptTestResult` | respondent segment, product concept, price, message, purchase intent, choice set, rejection reason | Needed for base purchase probability |
| `RepeatPurchaseProfile` | category, consumption cycle, satisfaction, repeat rate at 30/60/90 days, subscription propensity | Needed for repeat purchase and multi-year horizon |

Current marketplace feed placeholders:

- Market noise is randomly selected from a short hard-coded list.
- Competitor mentions are randomly selected with a 40% chance.
- Market noise appears with a 60% chance.
- Peer reviews and purchases come only from simulated neighbors, not external marketplace reviews.

Data needed to replace those:

| Feed element | Dataset |
|---|---|
| Category trend | Google Trends, marketplace search ranks, social listening, review velocity, media coverage |
| Competitor activity | Scraped/partner product catalog, price tracking, discounts, ad library, marketplace rank |
| Peer reviews | Real marketplace reviews, rating distribution, review text themes, review velocity |
| Peer purchases | Historical purchase/referral/cohort data or experiment observations |

### Phase 4: Decision Model And Propagation

Relevant files:

- `launchlens/phase4/loop.py`
- `launchlens/phase4/prompts.py`
- `launchlens/phase4/propagation.py`
- `launchlens/sim_lite.py`

Current mock decision formula:

```text
affordability = min(1.0, income / (price * 6.67))
social_boost = positive_peer_signals * 0.08 - negative_peer_signals * 0.05
influencer_trust = trusted_influencer_buy_signals * salience * 0.10
base_p = ISEC_BASE_BUY[isec] * affordability
p_buy = min(0.92, (base_p + social_boost + influencer_trust) * adoption_speed)
```

Hard-coded values:

| Parameter | Current value | Data required |
|---|---:|---|
| Base buy probability by ISEC | A1 55%, A2 45%, A3 38%, down to E3 0.5% | Category-specific purchase/trial model |
| Affordability cutoff | Price becomes fully affordable around 15% monthly HH income | Category and pack-size price elasticity |
| Positive social signal boost | +8 percentage points per positive signal | Referral/social proof experiment |
| Negative signal penalty | -5 percentage points per negative signal | Complaint/review sensitivity data |
| Trusted influencer boost | salience * 10 percentage points | Trust-source conjoint experiment |
| Adoption speed | innovator 2.0, early adopter 1.5, early majority 1.0, late majority 0.6, laggard 0.3 | Innovation adoption survey plus historical launch curves |
| Funnel advance | `p_buy * (0.3 + 0.2 * funnel_index)` | Observed awareness-to-research-to-consideration-to-purchase transition data |
| Rejection chance | +8% only when negatives exceed positives | Rejection benchmark by category |
| Post-buy share | 25% | Referral/share survey or campaign data |
| Post-buy complaint | 5% | Returns, reviews, complaint and low-rating data |
| Propagation decay | 0.70 per timestep/hop | Social sharing half-life / message recall data |
| Complaint boost | 1.5x | Negative WOM benchmark |

This is the most important missing data layer. Without it, the engine produces plausible adoption curves but not calibrated probabilities.

### Phase 5: Calibration

Relevant file:

- `NEXT_STEPS.md`
- `launchlens/phase5/__init__.py`

Phase 5 is effectively not implemented. The codebase has no `metrics.py`, no calibration fixtures, no historical launch JSONs, and no tests for calibration.

The planning doc asks for:

| Metric | Required ground truth |
|---|---|
| Overall adoption rate | Real trial/purchase/adoption rate by market and time window |
| Adoption curve shape | Weekly/monthly/quarterly time series after launch |
| Top segment accuracy | Actual buyers by age, income/SEC, gender, city, channel |
| Regional Spearman | City/district sales or adoption ranking |
| Rejection reason alignment | Survey, reviews, returns, complaints, social comments |

Historical launch case schema to create:

```json
{
  "case_id": "string",
  "brand": "string",
  "product_name": "string",
  "category": "string",
  "launch_date": "YYYY-MM-DD",
  "launch_geographies": ["district_or_city"],
  "price_mrp": 0,
  "launch_price": 0,
  "pack_size": "string",
  "positioning": "string",
  "distribution_channels": ["string"],
  "media_plan": [
    {
      "period": "YYYY-MM",
      "city": "string",
      "channel": "Meta|Google|TV|Retail|Influencer|Marketplace",
      "spend_inr": 0,
      "reach": 0,
      "impressions": 0,
      "clicks": 0,
      "conversions": 0
    }
  ],
  "sales_timeseries": [
    {
      "period": "YYYY-MM",
      "city": "string",
      "units": 0,
      "revenue_inr": 0,
      "new_buyers": 0,
      "repeat_buyers": 0
    }
  ],
  "buyer_segments": [
    {
      "segment": "string",
      "share": 0.0
    }
  ],
  "rejection_reasons": [
    {
      "reason": "string",
      "share": 0.0,
      "source": "survey|review|support|social"
    }
  ],
  "competitor_context": [
    {
      "brand": "string",
      "product": "string",
      "price": 0,
      "channel": "string",
      "rating": 0.0,
      "review_count": 0
    }
  ]
}
```

## Output Feasibility By Data Availability

| Desired output | Current code can output? | Defensible today? | Missing data |
|---|---:|---:|---|
| Purchase probability | Indirectly through simulated BUY rate | No | Base purchase model from surveys/transactions |
| Expected sales | No | No | Market size, distribution, awareness, conversion, inventory |
| Adoption curve | Yes | No | Historical launch time series and calibrated funnel transitions |
| City ranking | Only if multi-district profiles are supplied | No | City-level category demand, distribution, media, competition |
| Market-fit score | Not defined in code | No | Weighted metric definition and calibration labels |
| CAC | No | No | Spend, reach, click, conversion by channel/city/segment |
| Repeat purchase | Purchase history exists, but no repeat model | No | Consumption cycle, satisfaction, repeat cohort data |
| Rejection reasons | Yes from LLM/mock text | Weak | Survey/review/support reason taxonomy |
| Message resonance | Planned, not implemented | Weak | Concept tests and copy A/B results |

## Concrete Acquisition Plan

### Track A: Make The Existing Code Run With Real Population Data

Priority: immediate.

1. Build `data/raw/census/district_pca.csv`.
   - Source: Census India Primary Census Abstract, district level.
   - Required columns: district code, district/state name, total/male/female/urban/literate population.
   - Add age columns from C-13/C-14 where possible.

2. Build `data/raw/census/c16_language.csv`.
   - Source: Census C-16 mother tongue tables.
   - Shape: one row per district; language columns as counts or shares.
   - Normalize names to lowercase language keys used by persona name banks.

3. Build `data/raw/nfhs/nfhs5_district.csv`.
   - Source: NFHS-5 district factsheets.
   - Minimum column currently used: `mobile_internet_women`.
   - Add asset, sanitation, fuel, education, and internet variables for future wealth/tech modeling.

4. Generate `data/processed/districts/*.json`.
   - Run the existing `build-districts` CLI.
   - QA: `validate_population_diversity()` must pass for urban/rural, ISEC, and top languages.

### Track B: Replace Weak Demographic Proxies

Priority: high, because this drives who exists in the simulation.

1. ISEC/NCCS by district.
   - Best: MRSI/NCCS licensed dataset.
   - Practical fallback: construct from Census household assets, education, occupation, NSS/PLFS, NFHS wealth assets.
   - Output: `district_code`, A1-E3 shares.

2. Household income/expenditure by city/district.
   - Sources: NSS Consumer Expenditure Survey, CMIE Consumer Pyramids, ICE 360, primary survey.
   - Output: income/expenditure quantiles by district and urban/rural.

3. Occupation distribution.
   - Sources: Census workers table, PLFS, NSS.
   - Output: distribution across agriculture, manufacturing, trade/retail, formal services, informal services, homemaker, student, professional, self-employed.

4. Digital payment and access.
   - Sources: TRAI, IAMAI/Kantar, NPCI, payment partner data, primary survey.
   - Output: smartphone ownership, internet use, UPI active use by district, age, gender, income/SEC.

### Track C: Build Category Decision Dataset

Priority: critical for prediction.

Start with one wedge: skincare in Indian metros, because your example was a Rs. 499 skincare product in Bengaluru.

Recommended scope:

- Cities: Bengaluru, Mumbai, Delhi NCR, Hyderabad, Pune.
- Sample: 2,500-5,000 respondents.
- Categories: face serum, sunscreen, moisturizer, face wash, haircare, packaged beverage later as second vertical.

Survey modules:

| Module | Fields |
|---|---|
| Demographics | age, gender, city, pincode/district, household size, income band, education, occupation, language |
| Category usage | current use, frequency, spend, brand set, channel, purchase trigger |
| Price sensitivity | Van Westendorp or Gabor-Granger for Rs. 299-999 |
| Discrete choice/conjoint | product claims, brand type, price, pack size, ingredient, rating, channel, discount |
| Channel behavior | Amazon, Nykaa, quick commerce, D2C site, pharmacy, supermarket, local store |
| Social influence | reviews, influencers, friends/family, dermatologists, shopkeepers, WhatsApp |
| Purchase intent | likelihood, trial probability, expected timing |
| Rejection reasons | too expensive, distrust, ingredient concern, no need, existing brand loyalty, availability, low proof |
| Repeat purchase | expected usage cycle, satisfaction drivers, repurchase trigger |

Deliverable tables:

- `category_usage_city_segment.csv`
- `concept_test_responses.csv`
- `choice_model_long.csv`
- `price_sensitivity_curves.csv`
- `social_influence_weights.csv`
- `rejection_reason_taxonomy.csv`
- `repeat_purchase_benchmarks.csv`

These tables should replace `_ISEC_BASE_BUY`, affordability constants, adoption archetype speed, social boost, rejection, share, and complaint rates.

### Track D: Digital Market Signals

Priority: high for competitor context and live market feed.

Collect for each target category/city/channel:

| Dataset | Fields |
|---|---|
| Marketplace catalog | brand, product, category, pack size, MRP, selling price, discount, rating, review count, channel, availability |
| Review corpus | product, rating, review date, city if available, verified purchase flag, review text, helpful votes |
| Search/social trend | keyword, city/state, date, relative interest, platform |
| Influencer/social content | creator type, language, city, engagement, product/category mention, sentiment |
| Competitor promotions | brand, channel, date, campaign, discount, price, offer |

Sources:

- Amazon, Flipkart, Nykaa, BigBasket, Blinkit, Zepto catalog/reviews where legally permissible.
- Google Trends.
- Meta/TikTok/YouTube ad libraries and creator data if available.
- Partner brands and agencies for campaign data.

### Track E: Launch Experiment Data

Priority: highest for proprietary moat.

Run controlled tests with small D2C brands:

| Experiment | Data captured |
|---|---|
| Landing page concept test | city, segment, product concept, price, click, signup, waitlist, stated reason |
| Paid ad test | spend, impressions, reach, CTR, CPC, conversion, CAC by city/segment/creative |
| Marketplace soft launch | views, add-to-cart, purchase, rating, review, repeat order |
| Sampling campaign | sample request, redemption, purchase conversion, feedback |
| Repeat cohort | D30/D60/D90 repeat, churn, NPS, complaint/reason |

This is the dataset that can eventually let Sangati claim predictive value, not just plausibility.

### Track F: Historical Calibration Cases

Priority: required before selling accuracy.

Build at least 10 cases, but begin with 3:

1. Skincare: mamaearth Vitamin C Serum or a similar D2C skincare launch.
2. Beverage: Paper Boat Aam Panna or Raw Pressery.
3. Wearable/electronics: boAt Airdopes 141.

For each case, collect:

- launch date and launch cities
- price/MRP/pack size
- product claims and positioning
- channel availability by time
- monthly/quarterly units and revenue by region
- media spend by channel and geography
- segment profile of buyers
- review/rejection/complaint reasons
- competitor prices and campaigns
- repeat purchase/cohort behavior where applicable

Best sources:

- Paid: NielsenIQ/IRI/Kantar/Euromonitor/Statista/IDC, depending on category.
- Partner: brand Shopify/marketplace/Meta/Google/CRM data.
- Public: investor disclosures, interviews, marketplace reviews, app/product ranking histories, media reports.

## Suggested Repository Data Layout

```text
data/
  raw/
    census/
      district_pca.csv
      c16_language.csv
      age_distribution.csv
    nfhs/
      nfhs5_district.csv
    nss/
      household_expenditure.csv
    trai/
      telecom_penetration.csv
    npci/
      upi_adoption.csv
    market/
      catalog_prices.csv
      marketplace_reviews.parquet
      search_trends.csv
      competitor_promotions.csv
    surveys/
      concept_test_responses.csv
      choice_model_long.csv
      social_influence_weights.csv
      rejection_reason_taxonomy.csv
      repeat_purchase_benchmarks.csv
    calibration/
      cases/
        mamaearth_vitamin_c_serum.json
        paper_boat_aam_panna.json
        boat_airdopes_141.json
  processed/
    districts/
      <district_code>.json
    category_profiles/
      skincare_city_segment.csv
    model_params/
      purchase_propensity_skincare.json
      social_diffusion_weights.json
      repeat_purchase_skincare.json
```

## Code Changes Implied By The Data Plan

These are not required to inspect the repository, but they are needed to make the data useful.

1. Add `CategoryMarketProfile`, `MediaPlan`, `DistributionPlan`, `PriceBenchmark`, `LaunchCase`, and `CalibrationTarget` schemas.
2. Replace `_ISEC_BASE_BUY` with category-specific propensity parameters learned from survey/transaction data.
3. Replace fixed social boost and influencer trust constants with calibrated `social_influence_weights.csv`.
4. Replace random competitor mentions with real competitor feed rows.
5. Add expected sales calculation:
   - eligible population × awareness × channel availability × purchase probability × units per buyer.
6. Add CAC calculation:
   - media spend / acquired buyers by city/channel/segment.
7. Add repeat purchase:
   - first buyers × repeat probability by category/segment/time since purchase.
8. Implement `phase5/metrics.py` and calibration fixtures before claiming accuracy.

## Immediate 30-Day Plan

Week 1:

- Create Census PCA, language, NFHS files for the first 5 cities/districts.
- Generate real `DistrictProfile` JSONs.
- Validate persona diversity.

Week 2:

- Design and launch the skincare survey/conjoint for 2,500+ respondents.
- Build first competitor price/review dataset for skincare across Amazon/Nykaa/quick commerce.

Week 3:

- Fit first purchase propensity model by segment and price.
- Estimate social influence weights from survey responses.
- Replace mock constants in a config file rather than hard-coding them.

Week 4:

- Acquire or construct 1-3 historical launch cases.
- Implement Phase 5 metrics.
- Run calibration and report where the model is wrong.

## Bottom Line

The codebase is a good simulation scaffold, but the data gap is not just Census/NFHS. The missing core is category-specific behavioral ground truth: who buys, at what price, through which channel, after what exposure, under what social proof, and whether they repeat.

For an MVP, do not try to cover all consumer goods. Start with skincare in five metros, collect strong primary survey plus launch experiment data, and use public demographics only to build the synthetic population.
