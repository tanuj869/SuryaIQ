# SuryaIQ

**AI-augmented forecasting & policy decision-support for PM Surya Ghar rooftop-solar adoption.**

Capstone project. Two working layers:
1. **Core data-science layer** — ingest → clean → reconcile → merge →
   feature-engineer → models → scenarios.
2. **AI intelligence layer** — a cited, guardrailed, evaluated NL assistant
   (LangChain tools · LangGraph · Vectorless RAG · Guardrails · Evals). See
   [`ai/README.md`](ai/README.md).

Full spec: [`docs/SuryaIQ_Master_Build_Prompt.md`](docs/SuryaIQ_Master_Build_Prompt.md).

## What's built
- **36 States/UTs × 31 columns** canonical panel: `data/processed/master_state_panel.csv` / `.parquet`
- **Time-series panel** (only true series): `data/processed/re_capacity_panel.csv` — RE capacity, 36 states × FY2017-18→2024-25
- **Primary target:** `conversion_ratio` = installations / applications (range here ≈ 0.02–0.93 across states)
- **Models** (`src/models.py`): RE-capacity forecast (naive/linear/Holt) + RandomForest conversion model with leave-one-state-out CV
- **Subsidy-paradox test:** implementation-quality features (LOSO MAE 0.178) beat subsidy-amount features (0.194) — supports the hypothesis (associational)
- **Policy scenarios** (`src/scenarios.py`): A base · B subsidy +10% · C tariff +10% · D high-potential targeting · E combined
- **AI assistant** (`ai/`): lookup / compare / forecast / scenario / explain, every answer cited; out-of-scope queries rejected. Evals **8/8 pass**.
- Reconciliation, dataset register, and data dictionary under `docs/`

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src && python build_panel.py        # 1. build the data (from data/raw)
python paper_analysis.py               # 2. PAPER-FAITHFUL analysis -> paper_results.json (Tables I-IV)
python train.py                        # (aux) exploratory model set -> training_report.json
python scenarios.py                    # 3. policy scenarios
cd .. && python ai/cli.py              # 4. chat with the assistant (offline ok)
python ai/evals.py                     # 5. run the eval suite (17/17)
python ai/verify_report.py             # 6. verify every report feature (24/24)
./run_all.sh                           # 7. launch FastAPI + dashboard together
```

## Architecture (matches the AI Summary Report diagram)
```
User → Streamlit dashboard → FastAPI (api/main.py) → LangGraph (ai/graph.py)
     → tools / Vectorless RAG → Guardrails → structured response → Evals
```
- **FastAPI** (`api/main.py`): `uvicorn api.main:app --port 8000` — `/ask`, `/rank`,
  `/forecast/{state}`, `/scenario`, `/recommend`, `/search`, `/health`; docs at `/docs`.
  The dashboard's "Ask SuryaIQ" tab calls `/ask` (shows "Routed via FastAPI → LangGraph"),
  falling back to in-process LangGraph if the API is down.
- `./run_all.sh` starts both services together.

## Paper reproduction (authoritative analysis)
`src/paper_analysis.py` reproduces the IEEE paper *"Explaining State-Level Variation
in PM Surya Ghar Rooftop Solar Adoption: A Data-Driven Analysis of the Subsidy
Paradox"* (Gore & Sonawane) and writes `data/processed/paper_results.json`:
- **Table I** — covariate correlations (irradiance +0.56, tariff growth +0.42 strongest; subsidy/tariff *level* weak)
- **Table II** — **2-feature linear (LOO-CV MAE 0.145) beats 6-feature RandomForest (0.172)** — the subsidy-paradox headline
- **Table III** — log-linear national installation forecast (3/6/12-month)
- **Table IV** — scenario-support assessment (Scenario B subsidy-increase marked *unsupportable*)

The AI assistant, dashboard, and research report all report these paper-consistent
findings. (Reproduced numbers differ marginally from the paper — 0.145 vs 0.135 —
due to documented tariff-aggregation / DISCOM-exclusion choices; the ranking and
conclusions are identical.)

## Training (auxiliary model exploration)
`python src/train.py` trains and benchmarks the report's model set across every
dataset and writes `data/processed/training_report.json`:
- **Conversion regression** (target = conversion ratio): linear vs RandomForest vs
  gradient boosting (XGBoost stand-in — swap in XGBoost once `libomp` is available),
  leave-one-state-out CV. Best: RandomForest (MAE 0.172).
- **Top-quartile adopter classification**: RandomForest, 72% LOSO accuracy.
- **Time-series forecast** on the RE-capacity panel: naive / linear / Holt / **SARIMAX**, best-model per state.
- **Installation projection**: growth proxy for the report's "forecast installations".
Every one of the 11 raw files feeds a feature or target (see the Models & training dashboard tab).

## Dashboard
`streamlit run app/streamlit_app.py` → 5 views: **Overview** (KPIs, conversion &
installation rankings), **State explorer** (funnel + RE trajectory), **Drivers &
paradox** (potential-vs-capacity, tariff-vs-conversion, subsidy-paradox bars +
feature importance), **Scenario simulator** (subsidy/tariff sliders, live national
impact + targeting), and **Ask SuryaIQ** (the cited, guardrailed NL assistant).

## Pipeline
| Stage | File | Output |
|---|---|---|
| Config: paths, canonical state names, population | `src/config.py` | — |
| Per-file readers/cleaners | `src/ingest.py` | `data/interim/*.csv` |
| Merge + feature engineering + reconciliation | `src/build_panel.py` | `data/processed/*`, `docs/reconciliation_report.md` |
| Forecast + conversion model + subsidy-paradox | `src/models.py` | `data/processed/subsidy_paradox_results.json` |
| Policy scenario simulation | `src/scenarios.py` | scenario tables |
| AI assistant (tools·RAG·guardrails·graph·evals) | `ai/*` | `evals/results/eval_results.json` |

## Key design decisions (grounded in the data, not just the guideline)
- **Cross-section + short panel**, not long univariate forecasting — PMSG launched Feb 2024, so only File 10.2 (RE capacity) is a genuine multi-year series.
- **File 4 is canonical** for applications/installations/subsidy/capacity (most granular + recent); File 2 kept only for reconciliation.
- **Subsidy-paradox test** is the headline analysis: do implementation-quality features beat subsidy-amount features at predicting conversion? (see build prompt §4 Step 6).
- Scenarios are **associational, not causal** — label every scenario "predicted under model assumptions."

## Data freshness caveat
Sources are batch-published snapshots with **different dates** (Nov 2024 – Aug 2026).
National conversion computed here (~57%) reflects File 4's later snapshot, vs the
~34.5% an earlier snapshot showed in the literature. Always read with
`docs/dataset_register.md`.

## Next (from the build prompt)
EDA + case studies → time-series baselines on File 10.2 → tree-ensemble conversion
models + SHAP → policy scenarios → LangGraph/RAG/Guardrails/Evals assistant →
Streamlit dashboard → 4–6 page report.
