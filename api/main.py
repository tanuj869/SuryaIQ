"""SuryaIQ FastAPI service — the orchestration entry point in the architecture
diagram (User → Dashboard → FastAPI → LangGraph → tools/RAG → Guardrails →
Evals). Wraps the LangGraph assistant and the core tools as HTTP endpoints.

Run:  uvicorn api.main:app --port 8000   (from the repo root, venv active)
Docs: http://localhost:8000/docs
"""
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ai"))
sys.path.insert(0, str(ROOT / "src"))

import graph          # noqa: E402
import tools          # noqa: E402
import llm            # noqa: E402
import guardrails as G  # noqa: E402

app = FastAPI(title="SuryaIQ API",
              description="AI policy decision-support for PM Surya Ghar rooftop solar.",
              version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/", include_in_schema=False)
def root():
    """Redirect root requests to the Streamlit UI."""
    return RedirectResponse(url="http://localhost:8530")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    intent: str
    answer: str
    confidence: str
    data_source: list
    assumptions: list
    limitations: list
    forecast: Optional[dict] = None


@app.get("/health")
def health():
    prov = llm.provider()
    return {"status": "ok",
            "llm_mode": prov or "offline",
            "model": llm.model_for(prov) if prov else None,
            "states": tools._panel()["state"].nunique()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Main entry: routes a natural-language question through the LangGraph
    pipeline (guardrail → classify → tool → RAG → synthesize)."""
    if not req.question.strip():
        raise HTTPException(400, "question is empty")
    a = graph.ask(req.question)
    return AskResponse(question=a.question, intent=a.intent, answer=a.answer,
                       confidence=a.confidence, data_source=a.data_source,
                       assumptions=a.assumptions, limitations=a.limitations,
                       forecast=a.forecast)


@app.get("/state/{name}")
def state(name: str):
    r = tools.query_state(name)
    if "error" in r:
        raise HTTPException(404, r["error"])
    return r


@app.get("/rank")
def rank(metric: str = "installations", n: int = 5, ascending: bool = False):
    return tools.rank_states(metric, n=n, ascending=ascending)


@app.get("/national")
def national(query: Optional[str] = None):
    return tools.national_summary(query)


@app.get("/forecast/{state}")
def forecast(state: str, horizon: int = 3):
    return tools.run_forecast(state, horizon)


@app.get("/forecast-installations/{state}")
def forecast_installations(state: str, months: int = 12):
    return tools.run_forecast_installations(state, months)


@app.post("/scenario")
def scenario(subsidy_pct: float = 0.10, tariff_pct: float = 0.0):
    return tools.run_scenario(name="api", subsidy_pct=subsidy_pct, tariff_pct=tariff_pct)


@app.get("/recommend")
def recommend(n: int = 5):
    return tools.recommend(n=n)


@app.get("/search")
def search(q: str, k: int = 4):
    """Vectorless RAG retrieval endpoint."""
    return tools.summarize(q, k=k)
