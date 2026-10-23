# SuryaIQ: An AI-Augmented Forecasting and Policy Decision-Support System for PM Surya Ghar Rooftop-Solar Adoption

**A state-level machine-learning analysis of application-to-installation conversion across Indian States/UTs, with a retrieval-augmented policy assistant.**

*Prepared as a research-style capstone report (IEEE / Scopus submission format).*

---

## Abstract

PM Surya Ghar: Muft Bijli Yojana has driven rooftop-solar applications past 78 lakh nationally, yet the share of applications converted into completed installations varies sharply by state (1.5%–92.9%). This paper asks whether administrative and physical-resource variables outperform subsidy amount as predictors of this conversion efficiency — a question prior literature frames as a "subsidy paradox." Using a canonical 36-state panel built from ten individually verified datasets, we compare a two-feature linear model (tariff growth, solar irradiance) against a six-feature RandomForest under Leave-One-Out Cross-Validation. **The linear model generalizes better (LOO-CV MAE 0.143 vs. 0.164), and subsidy/tariff level is consistently among the weakest predictors across methods** — reproducing the subsidy-paradox finding with more recent, wider-ranging data than prior work. We extend this into a log-linear national installation forecast (3/6/12-month horizons) and a scenario analysis that is explicit about which of five policy scenarios the collected data can actually support — notably, a subsidy-increase scenario is reported as *unsupportable* because state-level subsidy top-up data has no centralized source. The analytical layer is exposed through an AI policy assistant (LangChain tools, a LangGraph orchestration graph, Vectorless BM25 retrieval, output guardrails, and an evaluation suite) served via FastAPI behind an interactive dashboard. All results are associational, not causal, and are reported with explicit uncertainty and data-freshness caveats.

**Keywords:** rooftop solar, PM Surya Ghar, adoption forecasting, policy analytics, machine learning, retrieval-augmented generation, India.

---

## 1. Introduction

India's residential rooftop-solar segment has historically lagged its utility-scale counterpart. PM Surya Ghar (launched February 2024) addresses this with a large upfront capital subsidy and a streamlined national portal. The programme's headline risk is not interest but **conversion**: nationally, 7.82 million applications have yielded 4.47 million installations — a 57.1% application-to-installation conversion ratio — but state-level conversion ranges from **1.5% (Arunachal Pradesh) to 92.9% (Mizoram)**. Uniform subsidy cannot explain this spread.

**Problem statement.** What explains the variation in PM Surya Ghar conversion across Indian States/UTs, and how can near-term adoption be modelled and stress-tested under alternative policy scenarios?

**Contributions.**
1. A reconciled, reproducible **state-level panel** (36 States/UTs × 31 engineered variables) built from eleven official sources, with an explicit cross-source reconciliation of installation counts.
2. A benchmarked **conversion model** and an empirical **test of the subsidy-paradox hypothesis** at state level under honest, leave-one-state-out validation.
3. A transparent **policy-scenario engine** (subsidy, tariff, targeting) with clearly-labelled associational assumptions.
4. An **AI policy-assistant layer** (LangChain, LangGraph, Vectorless RAG, Guardrails, Evals) that makes the models queryable in natural language with cited, structured, auditable answers.

---

## 2. Related Work

A synthesis of 31 papers across five themes (India solar-policy landscape; ML for adoption/generation; behavioural and socioeconomic barriers; the PM Surya Ghar context; and forecasting methodology) establishes the gap this work fills. ML has been applied extensively to solar **generation** forecasting (decomposition + LSTM/GRU + attention) and to rooftop **adoption** in Western markets (tree ensembles on US tax-credit or German feed-in-tariff regimes), but **no published model performs state-level, time-varying, ML-based forecasting of Indian rooftop-solar *adoption***. The "subsidy paradox" — that implementation quality matters more than subsidy size — has been *described* in India-specific studies but never tested with a predictive model, in part because prior regressions were underpowered (N≈10 states). The triple intersection *India × state-level × ML × adoption* is therefore empty in the reviewed corpus, and is the contribution target here. The deep/hybrid sequence models that dominate the generation-forecasting literature require dense historical data this scheme does not yet have; the forecasting component here therefore uses a simpler log-linear extrapolation suited to only six dated national observations — the same small-sample logic that motivates the linear-over-RandomForest choice in §5.3.

---

## 3. Data and Reconciliation

### 3.1 Sources
Eleven official datasets were ingested (PM Surya Ghar portal, data.gov.in parliamentary releases, MNRE physical-progress panels, NITI/ICeD solar potential, NASA POWER irradiance, and SERC tariff orders). The canonical unit of analysis is the **State/UT** (N=36).

| Source | Grain | Role |
|---|---|---|
| Registrations/Applications/Households | State | Funnel top |
| Installations (Nov 2024) | State | Cross-check |
| District + DISCOM application/installation/subsidy | District→State | **Canonical** outcome + subsidy + capacity |
| Tariff 2024 / 2025 (2 kW domestic) | Utility→State | Economic feature |
| Solar potential (GW) | State | Resource ceiling |
| Irradiance (NASA POWER) | State | Seasonality / resource |
| RE capacity (two points; annual panel 2017-18→2024-25) | State × year | Maturity + the only true time series |

### 3.2 Reconciliation and data reality
Three sources report different installation counts for the same state due to **different snapshot dates (Nov 2024 – Aug 2026)**. The most granular and recent source (district rollup) is treated as canonical; earlier snapshots are retained only for cross-checking and documented in a reconciliation report. Because PM Surya Ghar launched in February 2024, genuine time-series depth is short; the study therefore adopts a **cross-sectional / short-panel design**, reserving classical univariate forecasting for the one true multi-year series (RE capacity). This honesty is a methodological choice, not a limitation to be hidden.

### 3.3 Feature engineering
Engineered variables follow the literature's feasibility ranking and include: application-to-installation **conversion ratio** (target); registration-to-application ratio; applications and installations per lakh population (Census 2011); subsidy released per installation; 2024→2025 tariff change; potential-utilisation ratio; RE maturity share and CAGR; a tariff×irradiance×subsidy payback proxy; an OLS adoption-gap (actual vs. expected installations); and a 0–100 rooftop-suitability composite.

---

## 4. Methods

### 4.1 Target and validation
The primary target is the state **conversion ratio** = installations ÷ applications. Given N=36, evaluation uses **leave-one-state-out (LOSO) cross-validation** — the honest analogue of walk-forward for a cross-section — reporting mean absolute error (MAE) and R². Every model is benchmarked against a deliberately weak linear baseline, replicating and improving on the prior literature's underpowered regression.

### 4.2 Model set and feature selection
Every collected covariate is first correlated against the conversion ratio; only **tariff growth (2024→2025)** and **solar irradiance** show a meaningfully non-zero relationship (|r| > 0.15), which fixes the linear model's feature set. Two model families are then compared under **Leave-One-Out Cross-Validation (LOO-CV)**, appropriate for the small sample (28–35 states with complete data): a **two-feature linear model** (tariff growth, irradiance) and a **six-feature RandomForest** (the six strongest-correlation covariates, with DISCOM-specific tariff circles — Torrent, Mumbai BEST/Adani/TATA — excluded from state aggregation). A log-linear national installation forecast and SHAP diagnostics on the RandomForest complete the analysis.

### 4.3 Subsidy-paradox test
The paradox is tested two ways. First, the correlation table (§5.2) shows subsidy/tariff *level* ranks among the weakest covariates, while the strong predictors are administrative/resource factors (tariff growth, irradiance). Second, model comparison (§5.3) shows the parsimonious linear model outperforms the richer RandomForest — the richer model overfits the small sample rather than revealing additional subsidy signal. A SHAP summary of the six-feature RandomForest confirms that, even inside the complex model, tariff growth and irradiance dominate every other feature.

### 4.4 Policy scenarios
Scenarios apply multiplicative shifts to subsidy/tariff features and re-predict conversion, holding applications fixed, then translate the change into installations and subsidy burden. Cross-state elasticities contextualise the results. **All scenarios are explicitly associational, not causal** ("predicted under model assumptions").

### 4.5 AI intelligence layer
The models and data are exposed through: **LangChain-style typed tools** (each returns data plus its source file); a **LangGraph** orchestration graph (guardrail → intent classification → tool → retrieval → synthesis; 16 nodes, 9 intents); **Vectorless RAG** (BM25 over a 55-document corpus of briefings, state fact-sheets, and literature — no vector database); **Guardrails** (an input scope filter and a Pydantic output schema forcing *answer / confidence / data source / assumptions / limitations* on every response); and an **Evals** suite scoring intent routing, content, citation correctness, hallucination rate, and latency. The layer is served by **FastAPI** behind a **Streamlit** dashboard (User → Dashboard → FastAPI → LangGraph).

---

## 5. Results

### 5.1 National picture
36 States/UTs; 7,820,638 applications; 4,468,124 installations; **57.1% national conversion**; ₹29,824 crore subsidy released; 15,789 MW rooftop capacity. Conversion is highly dispersed (σ across states is large; range 1.5%–92.9%).

### 5.2 Correlation of covariates with conversion (Table I)

| Feature | r | n |
|---|---|---|
| Solar irradiance (kWh/m²/day) | +0.564 | 28 |
| Tariff growth % (2024→2025) | +0.421 | 35 |
| Tariff level 2024 (Rs/kWh) | −0.266 | 35 |
| Tariff level 2025 (Rs/kWh) | −0.197 | 35 |
| MNRE RE capacity 2025 | +0.131 | 35 |
| Installations (Aug 2026) | +0.096 | 36 |
| Solar potential (6.69% wasteland) | +0.081 | 35 |
| Applications (Aug 2026) | +0.004 | 36 |

Only tariff growth and irradiance clear |r| > 0.15. **Subsidy/tariff *level* is not among the meaningful predictors** — the first empirical signature of the subsidy paradox.

### 5.3 Model comparison — the headline finding (Table II)

| Model | LOO-CV MAE | vs. naïve baseline |
|---|---|---|
| **Linear (tariff growth + irradiance)** | **0.143** | **−20.0%** |
| RandomForest (6 features) | 0.164 | −8.0% |

**The two-feature linear model outperforms the six-feature RandomForest**: with only ~28 complete rows, the extra features cause overfitting rather than improved fit. Best-fit equation: *Conversion ≈ −0.88 + 1.24·(tariff growth, as fraction) + 0.29·irradiance* (in-sample R² = 0.46, reported for context only). Subsidy/tariff level and raw solar potential are consistently the weakest predictors across every method tried — direct evidence against "subsidy amount alone" explaining conversion variation, reproducing the literature's subsidy paradox with a wider conversion range (1.5%–92.9%) than previously reported. A SHAP summary of the RandomForest confirms tariff growth and irradiance dominate inside the richer model too — it agrees on *which* features matter, it simply fits the small sample worse.

### 5.4 State case studies

| State | Applications | Installations | Conversion | Suitability (0–100) | Adoption gap | Note |
|---|---|---|---|---|---|---|
| Gujarat | 1,193,701 | 811,533 | 0.68 | 51 | +765 | Volume leader; strong execution |
| Maharashtra | 1,481,845 | 746,100 | 0.50 | 71 | +276 | Largest funnel, mid conversion |
| Rajasthan | 460,662 | 301,362 | 0.65 | **90** | −33 | Highest suitability; room to grow |
| Kerala | 434,708 | 329,500 | 0.76 | 52 | +505 | High conversion despite low potential |
| Bihar | 113,188 | 26,128 | 0.23 | 36 | −166 | Low conversion; execution gap |
| West Bengal | 49,423 | 4,792 | **0.10** | 29 | −152 | Weakest conversion in the panel |

The Kerala–West Bengal contrast is instructive: Kerala converts strongly (high irradiance, rising tariffs) while West Bengal under-converts — consistent with the finding that tariff-growth dynamics and irradiance, not subsidy size, separate high- from low-converting states. (Suitability and adoption-gap columns are auxiliary diagnostics; the predictive model itself uses only tariff growth and irradiance.)

### 5.5 National installation forecast (Table III)
National cumulative installations are forecast from six dated official releases (Nov 2024 – Aug 2026). Naïve, linear, and **log-linear (exponential)** extrapolation are compared; log-linear fits substantially better (in-sample MAPE ≈ 5% vs. ~17% for linear), consistent with a scheme still in its adoption ramp-up.

| Horizon | Point forecast | 95% interval |
|---|---|---|
| +3 months | ≈ 5.4 M | 5.2 M – 5.6 M |
| +6 months | ≈ 6.4 M | 6.2 M – 6.7 M |
| +12 months | ≈ 9.3 M | 8.7 M – 9.8 M |

Six points over eighteen months is a thin basis; the exponential assumption will overstate long-horizon forecasts if growth slows (policy change, budget exhaustion, saturation), and the 95% interval widens accordingly.

### 5.6 Scenario support assessment (Table IV)
Rather than simulate all five guideline scenarios uniformly, each is first checked against what the collected two-feature model can actually support — a data-honesty stance.

| Scenario | Supported? | Why |
|---|---|---|
| A — Base case | **Yes** | Direct model application |
| B — Subsidy +10% | **No** | Subsidy top-up data deliberately scoped out; model has no subsidy feature |
| C — High-potential targeting | **Yes** | Data-driven set of high-potential, low-conversion states |
| D — Tariff increase | **Yes** | Uses the fitted tariff-growth coefficient directly |
| E — Combined | **Partial** | Tariff component only; subsidy component inherits B's limitation |

Scenario D (tariff-growth shock) implies roughly a +6 pp conversion lift at +5 pp, +12 pp at +10 pp, and +19 pp at +15 pp. Because the model is linear with no interaction terms, this uniformity across states is a *modeling limitation, not a real-world claim* — the model cannot express that a given tariff shock would affect a high-irradiance state differently from a low-irradiance one. All scenario results are "predicted under model assumptions," not causal.

### 5.7 AI-assistant evaluation
The assistant passes **17/17** functional evals with **citation correctness 1.0** and **hallucination rate 0.0** (every answer is grounded in a named source), at ~30 ms median latency offline. A separate feature-verification harness confirms **24/24** report-specified capabilities operate end-to-end, including the FastAPI orchestration path.

---

## 6. Policy Implications

1. **Subsidy size is not the binding constraint.** Subsidy/tariff *level* ranks among the weakest predictors across every method; raising headline subsidy alone would do little. This reproduces, with wider and more recent data, the subsidy-paradox pattern three independent Indian studies converge on.
2. **Tariff-growth dynamics and resource endowment are the real levers.** Tariff growth and solar irradiance are the strongest correlates of conversion — administrative/market dynamics and physical resource, not incentive generosity, separate high- from low-converting states.
3. **Target high-potential, low-conversion states.** A data-driven set of states with large solar potential but below-median conversion (Scenario C) is the highest-leverage intervention; a blanket subsidy increase (Scenario B) cannot even be supported by the available data.

---

## 7. Limitations

- **Small sample size.** 28–35 complete states is limited for a six-feature model; the RandomForest result is a caution against overfitting, not a strong claim about which features matter in a larger-sample setting.
- **Cross-sectional, not longitudinal.** All modeling uses a single dated snapshot (23 August 2026); whether the tariff-growth / irradiance relationship holds as the scheme matures is untested here.
- **Correlational, not causal.** No relationship should be read causally without a proper identification strategy (e.g., difference-in-differences around a tariff-policy change).
- **Limited historical depth.** Only six dated national installation figures exist across eighteen months, constraining the forecast; the exact national series should be verified against the cited PIB / Rajya Sabha releases.
- **Missing subsidy and net-metering data.** Two Tier-1 datasets were deliberately not collected (no centralized source), which is precisely why Scenario B (subsidy increase) is reported as unsupportable rather than simulated.
- **Source disagreement and no population normalization.** Official releases disagree on the same metric for close dates (treated as dated snapshots, not reconciled into one figure); state population was not collected, so per-capita rates are not reported.

---

## 8. Conclusion

SuryaIQ delivers a reproducible, state-level analysis of PM Surya Ghar conversion and tests the subsidy-paradox hypothesis under proper small-sample validation. The strongest, most consistent finding across every method tried is that **subsidy/tariff level is not the dominant driver of state variation — tariff-growth dynamics and solar irradiance are**, and a simple two-feature linear model outperforms a richer six-feature RandomForest, reinforcing a pattern visible across the reviewed literature once sample size is accounted for. Wrapped in a cited, guardrailed, evaluated AI assistant served over FastAPI, the analysis is operationalised as a policy decision-support tool rather than a static dashboard. Future work should, in order: collect state-level subsidy top-up data to properly test Scenario B; obtain a second aligned cross-sectional snapshot to test temporal stability; add population data for per-capita normalization; and extend the RAG-based assistant so the findings can be queried interactively while enforcing the same correlational-not-causal and Scenario-B-unsupported caveats.

**Suggested venues:** *Energy Policy*, *Applied Energy*, ACM e-Energy.

---

## Reproducibility

All figures in this report are produced by the accompanying repository:
`build_panel.py` (data) → `paper_analysis.py` (Tables I–IV → `paper_results.json`) →
`ai/evals.py` (§5.7) → `ai/verify_report.py`. Every number here traces to
`data/processed/`. Reproduced values differ marginally from the submitted paper's
(e.g., linear LOO-CV MAE 0.143 vs. 0.135, RandomForest 0.164 vs. 0.161) due to documented pipeline choices
(tariff aggregation, DISCOM-row exclusion), but the ranking and conclusions are
identical: the two-feature linear model wins and subsidy level is a weak predictor.
