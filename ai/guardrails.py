"""Guardrails: (1) input scope filter, (2) structured, auditable output schema.

Scope filter rejects out-of-scope / nonsensical / unsupported-prediction queries
BEFORE any tool runs. Output schema forces every answer to carry its forecast,
confidence, data source, assumptions, and limitations."""
import re
from typing import List, Optional

from pydantic import BaseModel, Field

# PMSG / rooftop-solar / policy vocabulary that keeps a query in scope
IN_SCOPE = re.compile(
    r"\b(solar|rooftop|pm surya|surya ghar|pmsg|install|adopt|convert|conversion|"
    r"subsidy|tariff|scheme|discom|state|forecast|predict|scenario|policy|"
    r"capacity|irradiance|potential|renewable|application|household|beneficiar|"
    r"grow|growth|report|evidence|recommend|summar|mnre|cea|niti|nasa|literature|"
    r"document|gap|paradox|target|rank|highest|lowest|national|average)",
    re.I)
# hard out-of-scope signals
OUT_SCOPE = re.compile(r"\b(mars|moon|stock|bitcoin|crypto|weather tomorrow|"
                       r"lottery|election winner|who will win|hack|bomb|weapon)\b", re.I)

INDIAN_STATES = None  # loaded lazily to avoid import cycle


def _states():
    global INDIAN_STATES
    if INDIAN_STATES is None:
        import pandas as pd
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "data/processed/master_state_panel.csv"
        INDIAN_STATES = set(pd.read_csv(p)["state"].str.lower())
    return INDIAN_STATES


def check_scope(query):
    """Return (allowed: bool, reason: str)."""
    q = query.strip()
    if len(q) < 3:
        return False, "Query too short."
    if OUT_SCOPE.search(q):
        return False, ("Out of scope: SuryaIQ only answers PM Surya Ghar "
                       "rooftop-solar adoption, forecasting, and policy questions.")
    # a bare non-Indian geography prediction request
    if re.search(r"\bpredict\b", q, re.I) and not IN_SCOPE.search(q):
        return False, "Unsupported prediction: outside the PM Surya Ghar domain."
    if IN_SCOPE.search(q) or any(s in q.lower() for s in _states()):
        return True, "in scope"
    return False, ("Out of scope: ask about PM Surya Ghar adoption, state "
                   "comparisons, forecasts, or policy scenarios.")


class SuryaIQAnswer(BaseModel):
    """Every assistant response conforms to this auditable structure."""
    question: str
    intent: str
    answer: str
    forecast: Optional[dict] = Field(default=None,
                                     description="forecast/scenario numbers if any")
    confidence: str = Field(description="high | medium | low")
    data_source: List[str] = Field(description="source files/models used")
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)

    def render(self):
        lines = [self.answer, "",
                 f"Confidence: {self.confidence}",
                 f"Data source: {', '.join(self.data_source)}"]
        if self.assumptions:
            lines.append("Assumptions: " + "; ".join(self.assumptions))
        if self.limitations:
            lines.append("Limitations: " + "; ".join(self.limitations))
        return "\n".join(lines)


DEFAULT_LIMITATIONS = [
    "PMSG launched Feb 2024 — short history; cross-section + short panel design.",
    "Sources are batch snapshots with differing dates (Nov 2024–Aug 2026).",
    "Scenario results are associational, not causal.",
]
