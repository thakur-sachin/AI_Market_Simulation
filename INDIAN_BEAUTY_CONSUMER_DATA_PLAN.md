# Indian Consumer Market Data Plan

Vertical focus: **Beauty & Personal Care**, starting with **masstige color cosmetics and entry skincare**.

Date: 2026-07-29

## 1. Decision

LaunchLens should not start with all FMCG. The first commercial wedge should be:

> Predict launch performance for Indian beauty and personal care brands, starting with color cosmetics and entry skincare across metro and Tier-1 women consumers.

Primary categories:

1. Lipstick / lip tint / lip gloss
2. Kajal / eyeliner / mascara
3. Foundation / compact / concealer
4. Sunscreen / face serum / moisturizer

Two operating modes:

1. **Customer pilot mode**: if Sugar Cosmetics, or another named brand, is the customer, use their first-party data under NDA and treat their products as confidential client scenarios.
2. **Market benchmark mode**: use licensed syndicated data, consumer survey data, and compliant marketplace/category signals to build neutral benchmarks without implying any named brand is a client or target.

Benchmark brand set should be handled generically in external material:

- `client_brand`
- `market_leader`
- `masstige_d2c_challenger`
- `value_challenger`
- `legacy_mass_brand`
- `premium_imported_brand`

Named brands can be used internally only when the data source is public, licensed, or provided by the brand under contract.

First geography:

- Bengaluru
- Mumbai
- Delhi NCR
- Hyderabad
- Pune

Second wave:

- Ahmedabad
- Chennai
- Kolkata
- Jaipur
- Lucknow
- Indore

## 2. Why This Vertical Wins

Beauty is the best first vertical because it has the right mix of data availability, emotional/social purchase behavior, frequent product launches, review text, price sensitivity, creator influence, and D2C/marketplace activity.

### Why Not Start With Broad FMCG

| Vertical | Problem |
|---|---|
| Staples: atta, rice, dal, edible oil | Too distribution-led, commodity-priced, low social/review richness |
| Household cleaning | Useful later, but lower influencer/review/social discovery signal |
| Packaged snacks/beverages | Good second vertical, but harder to get clean launch adoption and distribution/media data |
| Consumer electronics | Rich reviews but less FMCG-like repeat behavior |
| Beauty/personal care | Best first fit: high review volume, high social influence, strong D2C activity, survey availability |

### Why Beauty Fits LaunchLens

LaunchLens simulates personas, social influence, product messaging, price-value perception, and adoption over time. Beauty purchase decisions naturally depend on:

- Peer recommendation
- Influencer discovery
- Reviews and ratings
- Price tier
- Shade/skin compatibility
- Ingredient trust
- Brand credibility
- Occasion and identity
- Online/offline trial behavior
- Repeat purchase and switching

That makes beauty much better for an agent-based social simulation than commodity FMCG.

## 3. Evidence That Data Exists

These sources show that beauty has accessible market, panel, and consumer-survey data:

| Source | What It Gives | Why It Matters |
|---|---|---|
| Kantar Cosmetics Panel India | Monthly purchase tracking for face, eyes, lips, nails; 3,000+ women age 18-44; NCCS A/B; 10 lakh+ towns | Strongest syndicated panel fit for masstige color cosmetics |
| NIQ India beauty reports | Beauty trends, price tiers, reviews, social discovery, ingredient-led skincare, premiumization, Middle India | Category trend and benchmark source |
| Mintel India D2C Beauty Journey 2025 | Discovery touchpoints, online-first beauty purchase behavior, channel choices, new-age brand trust factors | Directly maps to LaunchLens decision prompts |
| Mintel India Beauty & Wellness / Clean Beauty | Wellness/clean/ingredient claims and consumer expectations | Useful for skincare and claim testing |
| YouGov India / YouGov Profiles | Audience intelligence, urban makeup-wearer data, online shopping and recommendation behavior | Useful for survey benchmarking and audience sizing |
| Marketplace reviews | Nykaa, Amazon, Flipkart, Myntra, Purplle, brand D2C sites | Rejection reasons, feature mentions, ratings, repeat complaints |
| Public demographics | Census, NFHS, TRAI, NPCI | Synthetic population, digital access, payment readiness |

Useful source links:

- Kantar India Cosmetics Panel: https://www.kantar.com/press-center/kantar-launches-its-first-ever-cosmetics-panel-in-india
- NIQ India Beauty 2025: https://nielseniq.com/global/en/insights/report/2025/the-new-face-of-indian-beauty/
- NIQ India Beauty 2026: https://nielseniq.com/global/en/insights/report/2026/indias-beauty-moment-growth-shifts-strategies-for-2026/
- Mintel India D2C Beauty Journey 2025: https://store.mintel.com/report/india-d2c-beauty-journey-market-report
- Mintel India Beauty & Wellness 2025: https://store.mintel.com/report/india-beauty-wellness-market-report
- Mintel India Reports: https://www.mintel.com/indian-consumer/
- YouGov makeup-wearers urban India article: https://yougov.com/articles/38412-makeup-media-consumers-india-survey
- Census PCA district data: https://censusindia.gov.in/nada/index.php/catalog/6191
- Census Population Finder: https://censusindia.gov.in/census.website/en/data/population-finder
- NFHS-5 district factsheet data: https://www.data.gov.in/catalog/national-family-health-survey-5-nfhs-5-india-districts-factsheet-data-provisional
- TRAI telecom subscription reports: https://www.trai.gov.in/release-publication/reports/telecom-subscriptions-reports
- NPCI UPI statistics: https://www.npci.org.in/product/upi/product-statistics
- NPCI UPI ecosystem/state-wise statistics: https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics

## 4. Product Scope For MVP

### MVP Product Type

Start with one hero simulation category:

> Lip product launch: long-stay lipstick / lip tint in the Rs. 299-799 range.

Why lip products first:

- Strong fit for a color-cosmetics customer pilot without needing to expose the customer's brand name.
- Kantar explicitly tracks lips as a category.
- Reviews are abundant.
- Purchase decision is easy for consumers to understand in a survey.
- Strong price sensitivity and shade/quality objections.
- Fast repeat/trial cycles compared to durables.
- Online and offline channels both matter.

### Second MVP Category

> Sunscreen or face serum in the Rs. 399-999 range.

Why second:

- Ingredient-led skincare is growing.
- NIQ and Mintel both cover ingredient transparency, efficacy, reviews, and digital discovery.
- Rich objections: irritation, trust, claims, price, results, skin type, ingredients.

## 5. Core Data Needed

LaunchLens needs six data layers.

### Layer 1: Population And Persona Data

Purpose: create realistic synthetic consumers.

| Dataset | Fields Needed | Where To Get | Priority |
|---|---|---|---|
| Census district profile | population, sex, age, urban/rural, literacy, households | Census India PCA, C-13/C-14, Population Finder | P0 |
| NFHS district indicators | mobile internet, phone ownership, education, sanitation, fuel, assets, health indicators | NFHS-5 district factsheets / data.gov.in | P0 |
| Telecom/digital access | wireless subscribers, broadband, rural/urban tele-density | TRAI reports | P1 |
| UPI/payment readiness | state-wise UPI volume/value, P2M transaction share | NPCI | P1 |
| Language | mother tongue/language distribution | Census C-16 | P1 |
| NCCS/ISEC proxy | education, occupation, durable ownership, household assets | Licensed MRSI/NCCS if possible; otherwise Census/NFHS proxy | P0 |

Output tables:

```text
data/raw/census/district_pca.csv
data/raw/census/c16_language.csv
data/raw/nfhs/nfhs5_district.csv
data/raw/trai/telecom_penetration.csv
data/raw/npci/upi_state_monthly.csv
data/processed/districts/<district_code>.json
```

### Layer 2: Category Market Data

Purpose: know who buys beauty, how often, at what price, through which channel.

| Dataset | Fields Needed | Where To Get | Priority |
|---|---|---|---|
| Cosmetics purchase panel | category, buyer segment, pack count, spend, online/offline, city/town tier, frequency | Kantar Cosmetics Panel India | P0 paid/vendor |
| Beauty market trends | price tiers, growth segments, reviews/social discovery, premiumization | NIQ India beauty reports | P0 paid/free report |
| D2C beauty behavior | discovery touchpoints, online-first brand trust, purchase channels, factors considered | Mintel India D2C Beauty Journey | P0 paid |
| Audience profiles | makeup users, online beauty shoppers, media consumption, recommendation sources | YouGov Profiles / custom survey | P1 paid |
| Category sizing | total market by category/subcategory/channel | NIQ, Kantar, Euromonitor, Statista, RedSeer, industry reports | P1 paid |

Output tables:

```text
data/raw/category/beauty_panel_purchase.csv
data/raw/category/beauty_category_size.csv
data/processed/category_profiles/color_cosmetics_city_segment.csv
data/processed/category_profiles/skincare_city_segment.csv
```

### Layer 3: Consumer Survey And Choice Data

Purpose: replace hard-coded buy probabilities with real behavior.

This is the most important proprietary dataset. We should run our own survey even if we buy reports.

#### Survey Sample

Minimum viable survey:

- 1,500 respondents
- Women 18-40
- NCCS A/B/C
- Bengaluru, Mumbai, Delhi NCR, Hyderabad, Pune
- At least 250-300 per city
- Must include online and offline beauty shoppers

Better survey:

- 3,000 respondents
- Women 16-44
- NCCS A/B/C/D
- 8-10 cities including Tier-2
- Oversample beauty buyers and recent category purchasers

Survey vendors/panels to evaluate:

- Kantar
- YouGov
- Toluna
- Dynata
- Cint
- Rakuten Insight
- Local Indian research agencies with beauty/FMCG panels

#### Survey Modules

| Module | Questions / Fields |
|---|---|
| Screening | city, age, gender, income/NCCS proxy, last 3-month beauty purchase, category usage |
| Beauty routine | categories used, frequency, monthly spend, current brands, online/offline split |
| Product trial behavior | willingness to try new brand, new shade, new ingredient, first-purchase trigger |
| Price sensitivity | acceptable price, expensive price, too expensive price, too cheap price; Gabor-Granger at Rs. 299/399/499/699/799 |
| Concept test | show 3-5 product concepts with brand/claims/price/channel; capture purchase intent |
| Discrete choice | choose among competitor products with price, claims, rating, discount, channel, influencer/review proof |
| Social influence | trust in friend, family, influencer, dermatologist, shopkeeper, marketplace review, brand ad |
| Channel behavior | Nykaa, Amazon, Flipkart, Myntra, Purplle, brand site, quick commerce, offline modern trade, local cosmetics store |
| Rejection reasons | too expensive, shade mismatch, brand distrust, bad ingredients, no need, poor reviews, delivery/authenticity concern |
| Repeat behavior | repurchase window, satisfaction drivers, switching reasons |

Output tables:

```text
data/raw/surveys/beauty_concept_test_responses.csv
data/raw/surveys/beauty_choice_model_long.csv
data/raw/surveys/beauty_price_sensitivity.csv
data/raw/surveys/beauty_social_influence_weights.csv
data/raw/surveys/beauty_rejection_reason_taxonomy.csv
data/raw/surveys/beauty_repeat_purchase_benchmarks.csv
```

### Layer 4: Marketplace And Review Data

Purpose: give product context, competitor benchmarks, ratings/reviews, objections.

Channels:

- Nykaa
- Amazon India
- Flipkart
- Myntra
- Purplle
- Tata Cliq Palette
- BigBasket / quick commerce where available
- Brand D2C sites

Collect for each SKU:

| Field | Use |
|---|---|
| brand | competitor mapping |
| product_name | product identity |
| category/subcategory | category benchmark |
| shade/variant | color cosmetics relevance |
| pack size | price comparability |
| MRP/selling price/discount | price sensitivity |
| rating/review_count | social proof |
| review text | objections and feature resonance |
| review date | launch/adoption curve proxy |
| verified purchase flag | quality filter |
| channel | channel benchmark |
| bestseller rank/search rank if available | demand proxy |
| availability/pincode if available | distribution proxy |

Output tables:

```text
data/raw/marketplace/beauty_catalog_prices.csv
data/raw/marketplace/beauty_reviews.parquet
data/raw/marketplace/beauty_review_themes.csv
data/raw/marketplace/beauty_competitor_price_tracking.csv
```

Important: scrape only where legally permissible and compliant with website terms. Prefer partner exports, vendor APIs, or licensed datasets when possible.

### Layer 5: Media, Influencer, And Discovery Data

Purpose: model awareness and social discovery.

| Dataset | Fields Needed | Where To Get |
|---|---|---|
| Ad spend and campaign data | channel, city, spend, impressions, reach, CTR, CPC, conversion | partner brand Meta/Google/marketplace dashboards |
| Influencer campaign data | creator, language, city, followers, engagement, content type, product mention, clicks, code redemptions | partner brands / influencer platforms |
| Search trends | keyword, city/state, time, relative interest | Google Trends |
| Social listening | mentions, sentiment, themes, creator type, language | social listening vendors / manual sample |
| Offline discovery | salon/store advisor, friends/family, retail trial | survey |

Output tables:

```text
data/raw/media/beauty_campaign_spend.csv
data/raw/media/beauty_influencer_posts.csv
data/raw/media/beauty_search_trends.csv
data/raw/media/beauty_social_listening.csv
```

### Layer 6: Historical Launch Calibration Cases

Purpose: prove the model predicts actual launches.

Start with 3 cases:

1. A customer-provided lip product / color cosmetics launch, if a pilot brand shares first-party data under NDA.
2. A public or licensed sunscreen/serum launch case from the entry-skincare market.
3. A public or licensed color-cosmetics launch case from a non-customer benchmark brand.

Do not use a prospective customer's brand name as a benchmark case unless the customer explicitly permits it. If Sugar Cosmetics is the pilot customer, its launch case should be private and labeled as `client_brand_case_001` or similar in shared documents.

For each case, collect:

| Field | Ideal Source | Fallback Source |
|---|---|---|
| launch date | brand partner | press release / marketplace first review date |
| price/MRP/discount | brand or marketplace | marketplace snapshots |
| product claims | product page | product page/archive |
| channel availability | brand/marketplace | manual tracking |
| weekly/monthly sales units | brand partner / NielsenIQ / Kantar | review velocity + rank proxy |
| buyers by city/channel/segment | brand/marketplace/CRM | survey + panel proxy |
| media spend and reach | brand dashboards | public campaign signal only |
| review/rejection reasons | marketplace reviews/support tickets | review corpus |
| repeat purchase | CRM/shopify/marketplace cohort | survey expected repeat |

Output case schema:

```json
{
  "product_id": "sugar_lip_tint_2025",
  "product_name": "Example Lip Tint",
  "category": "color_cosmetics_lip",
  "launch_date": "2025-08-01",
  "launch_geographies": ["Bengaluru", "Mumbai", "Delhi NCR", "Hyderabad", "Pune"],
  "price_mrp": 799,
  "launch_price": 599,
  "claims": ["long stay", "transfer proof", "hydrating"],
  "channels": ["Nykaa", "Amazon", "D2C", "offline"],
  "real_adoption_curve": [0.01, 0.025, 0.045, 0.07],
  "real_top3_segments": ["urban_A1-A3_18-24", "urban_B1-B2_25-34", "urban_A1-A3_25-34"],
  "real_top3_rejections": ["too expensive", "shade mismatch", "not enough proof"],
  "real_district_rates": {
    "Bengaluru": 0.08,
    "Mumbai": 0.07,
    "Delhi NCR": 0.06
  },
  "source_citations": []
}
```

## 6. How Each Dataset Plugs Into LaunchLens

| Data | Code/Model Use |
|---|---|
| Census/NFHS/TRAI/NPCI | `DistrictProfile`, persona sampling, digital readiness |
| Kantar/NIQ/Mintel category data | category priors, price tier benchmarks, channel assumptions |
| Survey/conjoint | purchase probability, price elasticity, social influence weights, rejection priors |
| Marketplace catalog | `ProductStimulus.competitor_context`, price benchmark, feature benchmark |
| Reviews | objection map, feature priority, rejection alignment, prompt context |
| Media plan | awareness probability and adoption timing |
| Influencer data | micro-influencer trust/reach parameters |
| Launch cases | Phase 5 calibration and model credibility |

## 7. New Schemas Needed

Add these after the combined branch is cleaned and merged:

```python
class CategoryMarketProfile:
    category: str
    geography: str
    buyer_penetration: float
    purchase_frequency_days: int
    average_spend_inr: int
    online_share: float
    offline_share: float
    premium_share: float
    seasonality_index: dict[str, float]

class PriceBenchmark:
    category: str
    brand: str
    product_name: str
    pack_size: str
    mrp: int
    selling_price: int
    rating: float
    review_count: int
    channel: str

class DistributionPlan:
    channel: str
    geography: str
    availability_share: float
    store_count: int | None
    pincode_coverage: float | None
    delivery_sla_hours: int | None

class MediaPlan:
    channel: str
    geography: str
    spend_inr: int
    impressions: int
    reach: int
    clicks: int
    conversions: int | None
    creative_variant: str

class LaunchCase:
    product_id: str
    launch_date: str
    geographies: list[str]
    real_adoption_curve: list[float]
    real_top3_segments: list[str]
    real_top3_rejections: list[str]
    media_plan: list[MediaPlan]
    sales_timeseries: list[dict]
```

## 8. Acquisition Plan

### Track A: Public Demographic Data

Owner: data/engineering

How to get:

1. Download Census PCA district file from Census India.
2. Download NFHS-5 district factsheet XLS from data.gov.in.
3. Download TRAI monthly telecom reports.
4. Download NPCI UPI monthly/state-wise statistics.
5. Normalize to city/district mapping for the first 5 cities.

Deliverable:

```text
data/processed/districts/
  BLR001.json
  MUM001.json
  DEL001.json
  HYD001.json
  PUN001.json
```

### Track B: Buy Or Access Syndicated Beauty Data

Owner: CEO/founder

Who to contact:

1. Kantar India: Cosmetics Panel / Worldpanel / Beauty & Fashion
2. NIQ India: Beauty reports and retail measurement
3. Mintel India: D2C Beauty Journey, Beauty & Wellness, Clean Beauty
4. YouGov India: Profiles / custom beauty audience cuts
5. Euromonitor / Statista / RedSeer only if budget allows

Ask for:

- Category penetration by city/tier
- Buyer demographics
- Channel split
- Price tiers
- Repeat frequency
- Brand shares
- New product launch readouts if available
- Cosmetics-specific panel extracts for lips/eyes/face/nails

Decision rule:

- If Kantar panel extract is affordable, buy it first.
- If not, buy Mintel/NIQ reports and run our own primary survey.

### Track C: Run Proprietary Consumer Survey

Owner: CEO + research consultant

How to get:

1. Write questionnaire.
2. Use YouGov/Toluna/Dynata/Cint/Rakuten Insight or Indian market research agency.
3. Target 1,500-3,000 qualified respondents.
4. Include concept test and choice/conjoint modules.
5. Export raw respondent-level CSV, not just a PDF summary.

Must-have raw columns:

```text
respondent_id
city
age
gender
income_band
nccs_proxy
education
occupation
primary_language
beauty_categories_used
monthly_beauty_spend
last_purchase_category
last_purchase_channel
brand_repertoire
price_acceptability
concept_purchase_intent
choice_task_selection
trusted_sources
rejection_reasons
repeat_purchase_intent
```

### Track D: Build Marketplace Dataset

Owner: data/engineering + legal review

How to get:

Preferred:

- Partner brand exports from Nykaa/Amazon/D2C dashboards.
- Licensed marketplace intelligence vendors.
- Manual sampled dataset for first 100-200 SKUs.

Fallback:

- Legally permissible public product-page sampling.
- Manual review collection for top competitor SKUs.

First SKU universe:

- 50 lip products
- 30 kajal/eyeliner products
- 30 foundation/compact products
- 50 sunscreen/serum products

Deliverable:

```text
data/raw/marketplace/beauty_catalog_prices.csv
data/raw/marketplace/beauty_reviews_sample.parquet
```

### Track E: Find Brand Partner

Owner: CEO

Best partner profile:

- D2C beauty/personal-care brand
- Launching a new SKU in next 1-3 months
- Has Shopify/marketplace/Meta/Google data
- Will share anonymized campaign/sales/review data
- Wants pre-launch concept testing and launch-readout report

Offer:

- Free or discounted pilot.
- Pre-launch simulation report.
- Post-launch calibration report.
- No resale of their raw data.
- Aggregated benchmark only with permission.

Data ask:

```text
daily_sales_by_channel
city_or_pincode_orders
ad_spend_by_channel
impressions_reach_clicks
discounts
reviews_and_ratings
returns_or_complaints
repeat_purchase_cohort
```

## 9. 30/60/90-Day Roadmap

### First 30 Days

Goal: make LaunchLens credible for one beauty use case.

1. Finalize first category: lip product launch.
2. Clean and merge `st/ML-Combined-Framework` as the working codebase.
3. Create first 5-city demographic profiles.
4. Buy/download at least one beauty report or panel extract.
5. Draft survey questionnaire.
6. Collect marketplace SKU/review sample for 100+ products.
7. Define LaunchCase schema and update calibration fixtures.

Output:

- `beauty_mvp_data_dictionary.md`
- first 5 district/city profiles
- first competitor dataset
- survey questionnaire ready to field

### Days 31-60

Goal: replace the weakest constants with data-backed priors.

1. Field 1,500-3,000 respondent survey.
2. Estimate price sensitivity curves.
3. Estimate social influence weights.
4. Build rejection reason taxonomy.
5. Fit first category propensity table by city/segment/price.
6. Create 1-3 historical launch cases using partner/public/proxy data.
7. Run calibration and document where model fails.

Output:

- `purchase_propensity_lip_products.json`
- `social_influence_weights_beauty.csv`
- `rejection_reason_taxonomy_beauty.csv`
- 1-3 real calibration case JSONs

### Days 61-90

Goal: produce a client-ready pilot.

1. Secure one D2C beauty brand pilot.
2. Run pre-launch simulation.
3. Compare 2-3 scenarios: price, claim, channel, influencer mix.
4. Deliver markdown/PDF report.
5. After launch, ingest actual sales/review/campaign data.
6. Calibrate and produce accuracy readout.

Output:

- client pilot report
- post-launch calibration report
- benchmark dataset for future clients

## 10. MVP Accuracy Claims

Do not claim:

- "We predict sales accurately."
- "We guarantee <8% adoption deviation."
- "We know exact CAC."
- "We forecast national FMCG demand."

Allowed early claims:

- "We simulate likely consumer response patterns."
- "We identify high-risk objections before launch."
- "We compare launch scenarios directionally."
- "We estimate which segments are most/least likely to adopt."
- "We calibrate against real launch data as it becomes available."

Target commercial claim after data collection:

> For Indian beauty launches, LaunchLens predicts relative segment adoption, top objections, and scenario ranking using synthetic consumers calibrated against survey, marketplace, and launch data.

## 11. Immediate CEO Checklist

This week:

- [ ] Choose first product use case: lip tint/lipstick or sunscreen.
- [ ] Contact Kantar India for Cosmetics Panel pricing/extract.
- [ ] Contact Mintel for D2C Beauty Journey report.
- [ ] Contact NIQ for India Beauty report/extract.
- [ ] Shortlist survey vendor: YouGov, Toluna, Dynata, Cint, Rakuten Insight, or local agency.
- [ ] Start outreach to 10 D2C beauty founders/growth heads for pilot data partnership.
- [ ] Ask engineering to make `st/ML-Combined-Framework` the cleaned base branch.

This month:

- [ ] Field survey.
- [ ] Build 5-city profiles.
- [ ] Build first marketplace review dataset.
- [ ] Build first LaunchCase JSON.
- [ ] Run one end-to-end calibrated simulation.

## 12. Bottom Line

Beauty and personal care is the right first Indian consumer vertical. Start with color cosmetics, especially lip products, because the survey data, panel data, social behavior, review data, and launch frequency are all favorable.

The CEO job is to get the data moat:

1. Syndicated category data.
2. Proprietary consumer survey/conjoint.
3. Marketplace/review data.
4. Brand partner launch data.
5. Historical calibration cases.

Once those exist, the engineering work in the feature branches can become a real commercial prediction product instead of only a plausible simulation demo.
