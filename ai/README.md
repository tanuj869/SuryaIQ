# SuryaIQ — AI Intelligence Layer

An AI policy-assistant layered over the core data-science foundation. Turns the
PM Surya Ghar dataset + models into a natural-language, cited, guardrailed,
evaluated assistant — exactly the stack specified in the project's AI Summary
Report (LangChain tools · LangGraph orchestration · Vectorless RAG · Guardrails ·
Evals).

## Run it

```bash
source .venv/bin/activate
python ai/cli.py                              # interactive
python ai/cli.py "What if subsidy rises 10%?" # one-shot
python ai/evals.py                            # run the eval suite
```

**Offline by default** — no API key needed; answers are deterministic,
tool-grounded, and fully cited.

### Turn on Claude polished-prose mode
The app auto-loads a **gitignored `.env`** file (your key never touches git or code):

```bash
cd suryaiq
cp .env.example .env         # then open .env and paste your key:
                            #   ANTHROPIC_API_KEY=sk-ant-...
python ai/check_llm.py       # confirms the key + model work
```

Then restart the dashboard so it picks up the key:
```bash
pkill -f "streamlit run app/streamlit_app.py"
streamlit run app/streamlit_app.py
```
The "Ask SuryaIQ" tab will show **Mode: Claude**. Same tools, same numbers, same
citations — Claude just rewrites the grounded draft into smoother prose. Default
model is `claude-sonnet-5` (override with `SURYAIQ_MODEL` in `.env`); it falls
back automatically if that id isn't on your plan.

## Architecture

```
question
  → guardrail_in        (ai/guardrails.py)  scope filter; reject out-of-domain
  → classify_intent     (ai/graph.py)       lookup | compare | forecast | scenario | explain
  → route → tool node   (ai/tools.py)       query_state / compare_states / run_forecast /
                                            run_scenario / subsidy_paradox
  → retrieve_context    (ai/retriever.py)   Vectorless RAG: BM25 over 49-doc corpus
  → synthesize          (ai/llm.py)         Claude (or offline template)
  → structure_output    (ai/guardrails.py)  Pydantic SuryaIQAnswer
```

Orchestrated as an explicit **LangGraph** `StateGraph` — each node is
independently testable.

## Components
| File | Role |
|---|---|
| `tools.py` | Typed, deterministic tools; each returns data **+ its source file**. Calls `src/models.py` and `src/scenarios.py`. |
| `retriever.py` | **Vectorless RAG** — BM25 + keyword over a 49-doc corpus (per-state fact sheets, project docs, 31-paper synthesis). No vector DB: free, fast, low-dependency. |
| `guardrails.py` | Input **scope filter** (rejects "predict installations for Mars", election/crypto/etc.) + Pydantic **output schema** forcing `answer / forecast / confidence / data_source / assumptions / limitations`. |
| `graph.py` | LangGraph workflow wiring all nodes. `ask(question) -> SuryaIQAnswer`. |
| `llm.py` | Claude wrapper with a key-free offline fallback. |
| `evals.py` + `../evals/test_cases.yaml` | Scores intent-routing, content checks, citation presence, scope rejection, latency. **8/8 pass, ~51 ms avg offline.** |
| `cli.py` | Interactive / one-shot terminal assistant. |

## What the assistant can answer (9 intents — full report Table-5 coverage)
- **Lookup** — "What is Gujarat's conversion ratio?" → 0.68 (cited)
- **Rank** — "Which state had the highest adoption?" → Gujarat; "top 5 by conversion", "lowest…", "underperforming"
- **Aggregate** — "National conversion rate?" / "How many states above 50%?" / "average tariff"
- **Compare** — "Compare Maharashtra and Gujarat" → ranks + numbers
- **Forecast** — "Forecast installations for the next 12 months" → installation projection; "forecast RE capacity" → SARIMAX/Holt
- **Scenario** — "What if subsidy rises 10%?" → +3.1% installs (predicted under model assumptions)
- **Explain** — "Why is adoption low in Bihar?" / "subsidy paradox" → data + LOSO evidence
- **Retrieve** — "Summarize the latest MNRE report" / "show supporting evidence" → cited passages (Vectorless RAG)
- **Recommend** — "Generate a policy recommendation" → prioritized high-potential, low-conversion states
- **Rejected** — out-of-scope queries ("predict installations for Mars") declined by the guardrail

Evals (`python ai/evals.py`): **17/17 pass**, citation-correctness 1.0, hallucination-rate 0.0
(the report's section-7 dimensions). In the dashboard text box, press **Enter** (or click away) to submit.

## Known limitation
The **ToD regime PDF (File 7) is image-based** — no extractable text, so it is
not in the RAG corpus yet. Add it via OCR (e.g. `ocrmypdf`/Tesseract) to enrich
the net-metering/ToD context, then drop the `.txt` pages into
`data/processed/corpus/` and rebuild the retriever.
