# SuryaIQ vs. ChatGPT — the real difference

> *"ChatGPT also answers these questions, so what's different?"*

A short, examiner-ready answer. ChatGPT is a **general-purpose language model**.
SuryaIQ is a **domain-specific decision-support *system*** that uses a language
model as **one small polish layer** over a grounded pipeline I built end-to-end.

## The 10 concrete differences

| # | ChatGPT (general LLM) | SuryaIQ (this project) |
|---|---|---|
| 1 | **No PM Surya Ghar data loaded.** Its training cutoff predates the Aug 2026 PMSG portal snapshot. | Loads 11 verified official sources (data.gov.in, MNRE, NITI, NASA POWER, SERC tariffs) into one reconciled 36-state panel. |
| 2 | **Cannot run ML on your data.** It cannot compute LOO-CV on a 36-row panel. | Trains the paper-faithful 2-feature linear model (MAE 0.143) and 6-feature RandomForest (MAE 0.164) under proper small-sample LOO-CV. |
| 3 | **Numbers come from "memory"** — statistical pattern completion over its training corpus; often stale or made up. | Numbers come from **deterministic tools** (`query_state`, `run_scenario`, `rank_states`); the LLM only phrases them. |
| 4 | **Hallucinates confidently** when it doesn't know. | Measured **hallucination rate 0.0 / citation correctness 1.0** across 17 eval cases, because every answer is anchored in a named source file. |
| 5 | **No citations.** You cannot trace an answer back to a specific file / row. | Every answer carries `data_source=[master_state_panel.csv, paper_results.json, …]` — auditable. |
| 6 | **No guardrails tied to your methodology.** Ask "what if subsidy +10%?" → it will invent a plausible-sounding number. | **Refuses that scenario** by design — because the paper's documented methodology says state-level subsidy top-up data has no centralized source. Honesty is enforced in code, not asked for in a prompt. |
| 7 | **No model explainability on your model.** It cannot run SHAP on your RandomForest. | SHAP decomposition per state, with global and local feature importance. (Maharashtra: tariff_2kw_2024 drags prediction by −0.063, annual_ghi lifts by +0.055.) |
| 8 | **Not reproducible.** Different users at different times get different answers. | **17/17 evals + 28/28 feature verification + 10/10 pytest**, deterministic, re-runnable on any machine. |
| 9 | **Not a research contribution.** You cannot publish "I used ChatGPT." | Reproduces an IEEE paper (Gore & Sonawane) with the same core finding — linear beats RandomForest, subsidy level weak — using documented tariff aggregation / DISCOM-exclusion choices. |
| 10 | **A chat window.** | A **deployable product**: FastAPI (`/ask`, `/scenario`, `/forecast`, `/search`), Streamlit dashboard with India map, SHAP view, scenario simulator, executive-brief generator, Docker + CI. |

## What my project actually trained / built (ChatGPT didn't)

- **Data pipeline:** `src/ingest.py` + `src/build_panel.py` — reads 11 Excel/PDF files, reconciles installation counts across different snapshot dates, handles DISCOM-vs-state double-counting, canonicalises 36 state-name variants.
- **Models:** `src/paper_analysis.py` + `src/explain.py` — trains Linear, RandomForest, SHAP decomposition under leave-one-state-out CV.
- **Scenario engine:** `src/scenarios.py` — applies the paper's tariff-growth coefficient to simulate state-scoped policy shocks.
- **AI orchestration:** `ai/graph.py` — a LangGraph `StateGraph` with 9 intents (lookup, rank, aggregate, compare, forecast, scenario, explain, retrieve, recommend, brief) and a guardrail layer.
- **Vectorless RAG:** `ai/retriever.py` — BM25 over a 55-doc corpus (briefings, state fact-sheets, literature synthesis). No vector DB.
- **Guardrails:** `ai/guardrails.py` — scope filter + Pydantic schema forcing `answer / confidence / data_source / assumptions / limitations` on every response.
- **Evaluation:** `ai/evals.py` + `ai/verify_report.py` + `tests/` — measures citation correctness, hallucination rate, intent accuracy, latency.
- **FastAPI service:** `api/main.py` — HTTP endpoints.
- **Dashboard:** `app/streamlit_app.py` — India choropleth, SHAP view, scenario simulator, executive-brief generator.
- **Reproducibility:** `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`.

## Try it: the acid test

Ask both systems the **same five questions** and compare. ChatGPT will fail every one of them in a way SuryaIQ cannot.

| Question | What ChatGPT does | What SuryaIQ does |
|---|---|---|
| "What is Maharashtra's PM Surya Ghar conversion ratio as of Aug 23, 2026?" | Guesses or refuses | **50.3%** (cited: master_state_panel.csv) |
| "Which 6 features matter most for conversion, by SHAP?" | Guesses | Returns the exact SHAP table from `shap_summary.json` |
| "What if subsidy increases by 10%?" | Confidently simulates it | **Refuses:** "Scenario B is NOT supportable — subsidy top-up data has no centralized source" |
| "What if tariff rises 10% in Maharastra?" (note typo) | Doesn't catch typo, invents a number | Fuzzy-matches to Maharashtra, applies paper's linear coefficient → 60.0% → 72.4%, +183,156 installations |
| "Give me an executive brief for Rajasthan" | Writes generic prose | SHAP-explained, scenario-tested, paper-cited 5-section memo |

## The LLM is ~5% of this project

The LLM only generates the **final sentence** from a structured tool result. Everything else — the data, the model, the guardrail, the evaluation, the deployment — is the student's engineering contribution. **ChatGPT could never pass the eval suite because ChatGPT doesn't have the data, the model, or the guardrail.** The whole point of the system is that *it would still work correctly in offline mode with no LLM at all* — try `python ai/cli.py` with `ANTHROPIC_API_KEY` unset. Every answer still comes out deterministic, cited, and correct.

**Bottom line:** this isn't a wrapper around ChatGPT. ChatGPT is the paintbrush; the system is the artist, the canvas, the paint, the gallery, and the auction catalogue.
