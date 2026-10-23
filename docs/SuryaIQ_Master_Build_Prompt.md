# SuryaIQ — Master Build Prompt
### PM Surya Ghar: AI-Augmented Rooftop-Solar Adoption Forecasting & Policy Decision-Support System

> Paste this whole document to an AI coding agent (Claude Code, Cursor, etc.) as the build brief. It is written to be executed end-to-end into a reproducible repository. It is grounded in the four source documents (guideline, AI summary report, SuryaIQ literature synthesis of 31 papers) and the 11 real data files provided. Where the data and the guideline disagree, the **data wins** — the honest constraints are called out explicitly so the project does not overclaim.

---

## 0. Role & Objective

You are a senior data scientist + ML engineer + energy-policy analyst. Build **SuryaIQ**: a research-grade system that (a) reconciles official PM Surya Ghar (PMSG) data into a clean state-level analytical dataset, (b) forecasts / models rooftop-solar **adoption and application→installation conversion** across Indian states, (c) runs transparent **policy scenarios**, and (d) exposes the results through an **AI policy-assistant dashboard** (LangGraph orchestration + Vectorless RAG + Guardrails + Evals).

Deliver it in **two layers** (both required):
1. **Core Data-Science Layer** — the analytically rigorous foundation.
2. **AI Intelligence Layer** — the orchestration/RAG/guardrails/evals assistant on top.

The output must be understandable to **both** a data-science audience and an energy-policy audience, and be submittable as a 4–6 page IEEE/Scopus-style research report.

---

## 1. Critical Data Reality (read before writing any code)

The provided files are mostly **cross-sectional snapshots**, not monthly time series. Honor this — do **not** fake a long monthly panel.

| # | File | Grain | Key columns | Role |
|---|------|-------|-------------|------|
| 1 | `Registrations_Applications_Households_2025-02-06.xlsx` | State × 1 snapshot (Feb 2025) | State/UT, Registration, Application, Households benefitted | Funnel top; conversion denominator |
| 2 | `Installations_27-11-2024.xlsx` | State × 1 snapshot (Nov 2024) | State/UT, Installation (Nos.) | Adoption outcome |
| 3 | `District-wise beneficiaries (Maharashtra)_11-12-2024.xlsx` | District (MH) × 1 | District, beneficiaries w/ CFA | Geo drilldown case study |
| 4 | `District_Discom_Application_Installation_Ratio_PMSG_...2026.xlsx` | **District & DISCOM** | State, District/Discom, Application, Installation, Household covered, Installation capacity (MW), **Subsidy released (lakhs)** | **Richest file** — primary source for conversion ratio, subsidy, capacity; roll up to state |
| 5.1 / 5.2 | `PMSG Tariff - 2024 / 2025.xlsx` | Utility × load slab (1/2/4/6/8/10 kW) | Utility, Tariff effective from, Av. Rate (P/kWh), Duty/Tax, Total | Tariff feature + payback economics; 2024→2025 gives one time delta |
| 6 | `Solar potential.xlsx` | State | Region, Potential @6.69% & @3% wasteland (GW) | Resource ceiling; utilization ratio |
| 7 | `ToD regime type.pdf` | State (document) | Net-metering / net-billing / ToD regime | Forward-looking policy feature (extract via RAG/manual coding) |
| 9 | `Solar irradiance.xlsx` | State × month climatology (NASA POWER) | GHI by month | Seasonality / resource suitability |
| 10.1 | `state-wise RE and non-RE installed capacity at two points.xlsx` | State × 2 points (2018, 2025) | Non-RE/RE at 31-Mar-2018 & 31-Mar-2025 | RE ecosystem maturity |
| 10.2 | `MNRE_RE_Capacity_AnnualPanel_2017-18_to_2024-25.xlsx` | **State × year (2017-18→2024-25)** | Annual RE capacity | **Only true time series** — use for the univariate/panel forecasting demo |

**Implications that must shape the design:**
- **Target variable:** application→installation **conversion ratio** = Installation / Application (per state), computed primarily from file 4 rolled up to state and cross-checked against files 1 & 2. Secondary targets: installations count, installed capacity (MW), households benefitted.
- **Design = state cross-section + short panel.** With ~28–36 states/UTs and PMSG live only since Feb 2024, favor **panel / cross-sectional ML (tree ensembles)** over long-horizon univariate forecasting. Reserve classic time-series forecasting (naive → Holt-Winters → ARIMA/SARIMAX) for file **10.2** (RE-capacity annual series) as the methodological baseline demonstration, and state it as such.
- **Reconciliation is a first-class task:** files 1, 2, and 4 report different installation numbers for the same states and different snapshot dates. Build an explicit reconciliation table; never silently pick one.
- **Honesty guardrails (from the synthesis):** do NOT claim "first ML on Indian solar" (generation forecasting exists); do NOT claim the subsidy paradox as original (it's documented); frame scenarios as "predicted under model assumptions," not causal, unless a defensible causal design is used.

---

## 2. Research Framing (lock these in)

- **Problem:** PMSG has a ₹75,000+ cr outlay yet application→installation conversion varies from ~1% to ~86% across states, unexplained by subsidy level alone. No state-level, time-varying ML model of Indian rooftop-solar *adoption* (not generation) exists.
- **Core questions:**
  1. Can state-level PMSG adoption/conversion be modeled with meaningful accuracy using ML?
  2. Do **implementation-quality** features (conversion history, subsidy-released efficiency, DISCOM/district dispersion) **outperform subsidy-amount** features? (the subsidy-paradox test)
  3. Which external factors (tariff, solar potential, RE maturity, irradiance, ToD regime) most associate with uptake?
  4. What happens under scenarios: subsidy +10%, tariff increase, high-potential-low-adoption targeting, combined?
- **Novelty claim (defensible):** the *combination* India + state-level + ML + **adoption/conversion** target + subsidy-paradox test + forward-looking ToD policy variable. Not novel on any single axis.

---

## 3. Repository & Deliverables

Create this structure:

```
suryaiq/
├── README.md                      # project overview, how to run, data-freshness notes
├── requirements.txt
├── data/
│   ├── raw/                        # the 11 original files, untouched
│   ├── interim/                    # per-file cleaned tables
│   └── processed/
│       ├── master_state_panel.parquet
│       ├── master_state_panel.csv
│       └── district_maharashtra.csv
├── docs/
│   ├── dataset_register.md         # per-file: source, publish/update date, grain, vars
│   ├── data_dictionary.md          # every column in the master table + units + transforms
│   └── reconciliation_report.md    # file1 vs file2 vs file4 installation discrepancies
├── notebooks/
│   ├── 01_data_inventory.ipynb
│   ├── 02_build_master_panel.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_forecasting_baselines.ipynb   # on file 10.2 series
│   ├── 05_panel_models_conversion.ipynb # tree ensembles + SHAP
│   └── 06_policy_scenarios.ipynb
├── src/
│   ├── ingest.py                  # readers/cleaners per file → interim
│   ├── build_panel.py             # merge → master_state_panel
│   ├── features.py                # engineered features
│   ├── models.py                  # baselines + ML + walk-forward eval
│   ├── scenarios.py               # elasticity + scenario simulation
│   └── config.py
├── ai/
│   ├── graph.py                   # LangGraph workflow definition
│   ├── tools.py                   # LangChain tools (query data, run forecast, run scenario, retrieve docs)
│   ├── retriever.py               # Vectorless RAG (BM25 + keyword + metadata over PDFs/CSVs)
│   ├── guardrails.py              # input scope + structured output schema
│   └── evals.py                   # test cases + metrics
├── app/
│   └── streamlit_app.py           # dashboard + AI assistant
├── evals/
│   ├── test_cases.yaml
│   └── results/
└── report/
    └── suryaiq_report.md          # 4–6 page research-style report
```

**Minimum deliverables (from the guideline):** dataset register + data dictionary; merged master parquet/CSV; reproducible pipeline; forecast comparison table; policy scenario table; interactive dashboard; 4–6 page report with uncertainty, limitations, ≥5 state case studies + 1 national result.

---

## 4. Core Data-Science Layer — step by step

### Step 1 — Data inventory & register
- Read all 11 files. For each, record in `dataset_register.md`: source portal, publication date, snapshot date, grain, variables, units, known caveats. Note file 9's irradiance sheet has blank leading rows — parse robustly.

### Step 2 — Clean per file (`ingest.py`)
- Standardize `State/UT` names to one canonical list (map "Andaman & Nicobar Island", "LEH LADAKH" casing, Delhi/NCT, Daman & Diu / DNH merges, etc.). Keep a `state_name_map`.
- Coerce `NA` → missing; keep a missingness log.
- Tariff files: pick the **2 kW domestic** slab as the representative residential tariff (typical PMSG household), keep others available; convert Paise/kWh → ₹/kWh; compute 2024→2025 tariff change.
- File 4: roll **district → state** (sum applications, installations, households, capacity, subsidy released). Keep district & DISCOM granularity for drilldowns.
- File 10.2: parse the regional headers (North/West/etc.) into a tidy State × Year long table.

### Step 3 — Build canonical state table (`build_panel.py`)
One row per State/UT with (at minimum):
`state, registrations, applications, installations, households_benefitted, installed_capacity_mw, subsidy_released_cr, avg_tariff_2kw_2024, avg_tariff_2kw_2025, tariff_change_pct, solar_potential_gw, annual_ghi, re_capacity_2018, re_capacity_2025, re_cagr, tod_regime, source_files, snapshot_dates`.
Keep file 10.2 as a separate tidy `re_capacity_panel` (State × Year) for time-series work.

### Step 4 — Feature engineering (`features.py`)
From the synthesis's **Feature Feasibility Matrix**, implement the **"Strongly Recommended"** tier first:
- `conversion_ratio = installations / applications` ← **primary target**
- `reg_to_app_ratio = applications / registrations`
- `app_per_lakh_pop`, `install_per_lakh_pop` (bring state population; Census/MoSPI — document the source)
- `subsidy_per_installation = subsidy_released / installations`
- `potential_utilization = installed_capacity_mw / (solar_potential_gw*1000)`
- `re_maturity = re_capacity_2025 / total_capacity_2025`
- `tariff_payback_proxy = f(tariff, subsidy, irradiance)` (document the formula)
- Categoricals: `tod_regime`, `region`
- Seasonality (for file 9 / 10.2 series): month, quarter, summer/monsoon indicators.
Flag "awareness" as a **known unmeasured gap** — omit or use an explicit proxy, and say so.

### Step 5 — EDA (`03_eda.ipynb`)
State rank tables; **funnel** (registration→application→installation→households); conversion-ratio choropleth; tariff vs conversion scatter; potential vs adoption (adoption-gap) scatter; irradiance seasonality; missingness/reporting-break report. Produce the ≥5 state case studies here (e.g., Gujarat, Maharashtra, Rajasthan, Bihar, Kerala + one high-potential-low-adoption state).

### Step 6 — Modeling
- **Time-series baselines on file 10.2** (RE-capacity annual series, national + ≥5 states): naive, seasonal naive, Holt-Winters, ARIMA/SARIMA. **Walk-forward / rolling-origin** validation. Report MAE, RMSE, MAPE/sMAPE vs the naive baseline. This satisfies the guideline's "baselines before advanced models."
- **Cross-sectional / panel ML for conversion** (primary contribution): Random Forest + **XGBoost/LightGBM** predicting `conversion_ratio` (regression) and a **classification** variant (will a state be top-quartile adopter?). Given small N, use **leave-one-state-out / k-fold CV**, repeat with seeds, report mean±sd. Baselines: linear/logistic regression (deliberately replicating P2's weak baseline) + single decision tree.
- **Explainability:** SHAP feature importance. **Explicitly test the subsidy-paradox** — compare a subsidy-amount-only model vs an implementation-quality model vs combined; report which wins.
- Produce a single **forecast/model comparison table** (`report`).

### Step 7 — Policy scenarios (`scenarios.py`)
Estimate elasticities from cross-state variation (clearly labeled associational, not causal). Simulate and tabulate installations / households / subsidy burden under:
- **A Base** (observed) · **B Subsidy +10%** · **C Tariff increase** · **D High-potential-low-adoption targeting** · **E Combined (B+D)**.
Every scenario output must carry "predicted under model assumptions" + uncertainty band. Build a **Rooftop Suitability Index** (composite of solar potential, tariff, RE maturity, conversion) and an **adoption-gap** ranking (actual vs expected given potential+tariff+income).

### Step 8 — Validation & reporting standards
Walk-forward for series; repeated CV for panel; every model beaten against naive/linear baseline; prediction intervals shown; **data-freshness disclaimer** prominent (snapshots differ by date; PMSG updates in batches, no clean real-time API).

---

## 5. AI Intelligence Layer

Build the assistant exactly as the AI Summary Report specifies, using the **recommended lightweight stack**.

### 5.1 LangChain tools (`ai/tools.py`)
Expose typed tools: `query_state_data(state, metric)`, `compare_states(a, b)`, `run_forecast(state, horizon)`, `run_scenario(name, params)`, `retrieve_docs(query)`. Each returns structured data + the source file(s) used.

### 5.2 LangGraph orchestration (`ai/graph.py`)
Explicit workflow graph (not one monolith), e.g. for "What if subsidy +10%?":
`classify_intent → retrieve_data → run_forecast → run_scenario → compare_to_baseline → generate_explanation → attach_visual`. Each node independently testable.

### 5.3 Vectorless RAG (`ai/retriever.py`) — **recommended over a vector DB**
Corpus = the provided PDFs (ToD regime, guideline), CSVs, and any added MNRE/CEA/NITI policy PDFs. Retrieval via **BM25 + keyword match + metadata filtering + SQL over the processed tables**. Rationale (state it in README): easier, free, faster, fewer dependencies for a ~20-PDF / ~15-CSV corpus. Every answer must cite the retrieved source.

### 5.4 Guardrails (`ai/guardrails.py`)
- **Input scope filter:** accept forecast / policy / state-comparison questions; reject out-of-scope ("predict installations for Mars"), illegal, or unsupported-prediction requests with a clear refusal.
- **Structured output schema:** every assistant answer returns `{ answer, forecast, confidence, data_source, assumptions, limitations }` — auditable, not free text.

### 5.5 Evals (`ai/evals.py`, `evals/test_cases.yaml`)
Test both layers. Golden test cases (expected vs actual), plus dimensions: forecast accuracy (RMSE/MAPE), hallucination rate, citation correctness, response quality, latency, cost. Emit a results table into `evals/results/`.

### 5.6 Dashboard (`app/streamlit_app.py`)
Streamlit app with: national overview, state explorer (funnel, choropleth, rank), forecast viewer with intervals, scenario simulator (sliders for subsidy/tariff), adoption-gap & suitability-index views, and a **natural-language assistant panel** wired to the LangGraph pipeline. Every AI answer shows its citations + assumptions + limitations block.

---

## 6. Architecture (end-to-end)

```
User → Streamlit Dashboard → FastAPI (optional) → LangGraph
        → [LangChain tools: data / forecast / scenario / Vectorless-RAG retriever]
        → Guardrails (scope in, structure out) → Final Response → Evals (logged)
```

---

## 7. Report (`report/suryaiq_report.md`, 4–6 pages, research style)

Sections: Abstract · Introduction & problem · Related work (use the 31-paper synthesis; cite the five sections) · Data & reconciliation (with freshness caveats) · Methods (baselines + panel ML + SHAP + scenarios + AI layer) · Results (model comparison table, subsidy-paradox finding, ≥5 state case studies + national result, scenario table) · Policy implications · Limitations (short history, small N, awareness gap, snapshot reconciliation, associational-not-causal) · Conclusion. Suggested venues: Energy Policy, Applied Energy, ACM e-Energy.

---

## 8. Execution order (do in this sequence)

1. Scaffold repo + `requirements.txt` (pandas, numpy, openpyxl, scikit-learn, statsmodels, xgboost, lightgbm, shap, matplotlib/plotly, rank-bm25, langchain, langgraph, streamlit, pydantic).
2. Ingest + register + data dictionary + **reconciliation report** (do this before any modeling — the guideline is explicit).
3. Build master state panel + features → `processed/`.
4. EDA + state case studies.
5. Time-series baselines on file 10.2.
6. Panel ML for conversion + SHAP + subsidy-paradox test + model comparison table.
7. Scenarios + suitability index + adoption-gap.
8. AI layer: tools → retriever → graph → guardrails → evals.
9. Streamlit dashboard wiring both layers.
10. Report + README + repo cleanup.

## 9. Non-negotiable quality rules
- Reproducible: fixed seeds, `requirements.txt`, run instructions in README.
- Nothing fabricated: if population/income/ToD data is added, cite the real source in the dataset register; if a number can't be sourced, mark it `TODO: verify` rather than inventing it.
- Keep predictive scenarios separate from causal claims, in both code comments and the report.
- Every AI answer is cited, scoped, and structured.
```
```
