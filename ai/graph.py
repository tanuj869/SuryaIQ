"""LangGraph orchestration of the SuryaIQ assistant.

Workflow:  guardrail_in -> classify_intent -> route -> (tool node) ->
           retrieve_context -> synthesize -> structure_output

Each stage is an explicit node, independently testable. Falls back cleanly
when no API key is present (see ai/llm.py)."""
import re
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END

import llm
import tools
import guardrails as G


class S(TypedDict, total=False):
    question: str
    allowed: bool
    reject_reason: str
    intent: str
    params: dict
    tool_result: dict
    rag_hits: list
    sources: list
    answer: G.SuryaIQAnswer


# ------------------------------------------------------------------ nodes
def guardrail_in(s: S) -> S:
    ok, reason = G.check_scope(s["question"])
    return {"allowed": ok, "reject_reason": reason}


def classify_intent(s: S) -> S:
    q = s["question"].lower()
    params = {}
    is_change = re.search(r"increase|decreas|\brise|raise|cut|boost|double|hike|"
                          r"\+?\d+\s*%|higher|lower", q)
    if re.search(r"\bwhat if\b|\bscenario\b", q) or \
            (re.search(r"subsid|tariff", q) and is_change):
        intent = "scenario"
        params["subsidy_pct"] = 0.10 if "subsid" in q else 0.0
        params["tariff_pct"] = 0.10 if "tariff" in q else 0.0
    elif re.search(r"\b(brief|executive brief|policy brief|one[- ]pager|"
                   r"memo|dossier)\b", q):
        intent = "brief"
    elif re.search(r"\bsummar|show (supporting )?evidence|what does.*(report|say)|"
                   r"according to|cite|literature|mnre report|cea report", q):
        intent = "retrieve"
    elif re.search(r"\brecommend|policy recommendation|which states? (to|should).*"
                   r"(target|prioriti|focus)|what should.*(do|government)|priorit", q):
        intent = "recommend"
    elif re.search(r"\bforecast|predict|next \d+|projection|trajectory|future", q):
        intent = "forecast"
    elif re.search(r"\bhighest|lowest|top \d+|bottom \d+|most|least|best|worst|"
                   r"rank|which state|underperform|leader|slowest|fastest", q):
        intent = "rank"
    elif re.search(r"\bnational|overall|in total|total number|across india|average|"
                   r"how many states|mean\b|sum of", q):
        intent = "aggregate"
    elif re.search(r"\bcompare|versus| vs |better than|difference between", q):
        intent = "compare"
    elif re.search(r"\bwhy|explain|reason|because|driver|factor|paradox", q):
        intent = "explain"
    else:
        intent = "lookup"
    return {"intent": intent, "params": params}


def _states_in(q):
    import difflib
    hits = []
    ql = q.lower()
    states = list(tools._panel()["state"])
    # exact substring match first
    for s in states:
        if s.lower() in ql:
            hits.append(s)
    if hits:
        return hits
    # fuzzy fallback — catch common misspellings like "maharastra" / "tamilnadu"
    for word in re.findall(r"[A-Za-z][A-Za-z &]{3,}", q):
        wl = word.strip().lower()
        if wl in ("the", "and", "for", "what", "will", "happen", "scenario",
                  "subsidy", "tariff", "increase", "decrease", "forecast",
                  "predict", "which", "state", "states", "india", "national"):
            continue
        m = difflib.get_close_matches(wl, [s.lower() for s in states], n=1, cutoff=0.82)
        if m:
            match = next(s for s in states if s.lower() == m[0])
            if match not in hits:
                hits.append(match)
    return hits


def route(s: S) -> str:
    if not s["allowed"]:
        return "rejected"
    return s["intent"]


def node_lookup(s: S) -> S:
    sts = _states_in(s["question"])
    # no named state → answer with the national summary instead of erroring
    res = tools.query_state(sts[0]) if sts else tools.national_summary(s["question"])
    return {"tool_result": res, "sources": [res.get("source", "")]}


def node_rank(s: S) -> S:
    q = s["question"].lower()
    metric = tools.detect_metric(q)
    asc = bool(re.search(r"lowest|least|worst|bottom|underperform|slowest", q))
    if "underperform" in q:
        metric, asc = "adoption_gap", True
    m = re.search(r"top (\d+)|bottom (\d+)", q)
    n = int(next(g for g in m.groups() if g)) if m else 5
    res = tools.rank_states(metric, n=n, ascending=asc)
    return {"tool_result": res, "sources": [res["source"]]}


def node_aggregate(s: S) -> S:
    res = tools.national_summary(s["question"])
    return {"tool_result": res, "sources": [res["source"]]}


def node_compare(s: S) -> S:
    sts = _states_in(s["question"])
    if len(sts) >= 2:
        res = tools.compare_states(sts[0], sts[1])
    else:
        res = {"error": "need two states to compare"}
    return {"tool_result": res, "sources": [res.get("source", "")]}


def node_forecast(s: S) -> S:
    q = s["question"].lower()
    sts = _states_in(s["question"])
    # "forecast installations" -> installation projection; else RE-capacity series
    if re.search(r"install", q):
        m = re.search(r"next (\d+)\s*month|(\d+)\s*month", q)
        months = int(next(g for g in m.groups() if g)) if m else 12
        # No state named -> national installation forecast (Table III), not a
        # state default. A state named -> that state's installation projection.
        if sts:
            res = tools.run_forecast_installations(sts[0], months)
        else:
            res = tools.national_forecast(months)
    else:
        state = sts[0] if sts else "Maharashtra"
        m = re.search(r"next (\d+)", q)
        h = min(int(m.group(1)), 5) if m else 3
        res = tools.run_forecast(state, h)
    return {"tool_result": res, "sources": [res.get("source", "")]}


def node_retrieve(s: S) -> S:
    res = tools.summarize(s["question"], k=4)
    return {"tool_result": res, "sources": res.get("sources", [])}


def node_recommend(s: S) -> S:
    res = tools.recommend(n=5)
    return {"tool_result": res, "sources": [res.get("source", "")]}


def node_brief(s: S) -> S:
    sts = _states_in(s["question"])
    state = sts[0] if sts else "Maharashtra"
    res = tools.state_brief(state)
    return {"tool_result": res, "sources": res.get("sources", [])}


def node_scenario(s: S) -> S:
    p = s.get("params", {})
    sts = _states_in(s["question"])  # scope to a named state if present
    res = tools.run_scenario(name="query", subsidy_pct=p.get("subsidy_pct", 0.10),
                             tariff_pct=p.get("tariff_pct", 0.0),
                             target_states=sts or None)
    return {"tool_result": res, "sources": [res.get("source", "")]}


def node_explain(s: S) -> S:
    res = {}
    if "paradox" in s["question"].lower() or "subsidy" in s["question"].lower():
        res["paradox"] = tools.subsidy_paradox()
    sts = _states_in(s["question"])
    res["states"] = {st: tools.query_state(st) for st in sts[:2]}
    srcs = ["subsidy_paradox_results.json", "master_state_panel.csv"]
    return {"tool_result": res, "sources": srcs}


def retrieve_context(s: S) -> S:
    # pull supporting passages only for intents that benefit from them
    rag_hits = []
    if s["intent"] in ("explain", "retrieve", "lookup", "recommend"):
        rag_hits = tools.retrieve_docs(s["question"], k=3)["hits"]
    srcs = list(dict.fromkeys(s.get("sources", []) + [h["source"] for h in rag_hits]))
    return {"rag_hits": rag_hits, "sources": [x for x in srcs if x]}


def synthesize(s: S) -> S:
    tr = s.get("tool_result", {})
    answer_text = llm.narrate(s["question"], s["intent"], tr, s.get("rag_hits"))
    forecast = tr if s["intent"] in ("forecast", "scenario", "recommend") else None
    conf = {"lookup": "high", "compare": "high", "rank": "high",
            "aggregate": "high", "retrieve": "medium", "recommend": "medium",
            "brief": "medium", "explain": "medium", "forecast": "low",
            "scenario": "low"}.get(s["intent"], "medium")
    assumptions = []
    if s["intent"] == "scenario":
        assumptions = ["Applications held fixed; only conversion responds.",
                       "Elasticity inferred from cross-state variation."]
    ans = G.SuryaIQAnswer(
        question=s["question"], intent=s["intent"], answer=answer_text,
        forecast=forecast, confidence=conf, data_source=s["sources"],
        assumptions=assumptions, limitations=G.DEFAULT_LIMITATIONS)
    return {"answer": ans}


def node_rejected(s: S) -> S:
    ans = G.SuryaIQAnswer(
        question=s["question"], intent="rejected",
        answer="Rejected by guardrail: " + s["reject_reason"],
        confidence="high", data_source=["guardrail"],
        limitations=["Out-of-scope queries are declined by design."])
    return {"answer": ans}


# ------------------------------------------------------------------ build
def build_graph():
    g = StateGraph(S)
    for name, fn in [("guardrail_in", guardrail_in),
                     ("classify_intent", classify_intent),
                     ("lookup", node_lookup), ("compare", node_compare),
                     ("forecast", node_forecast), ("scenario", node_scenario),
                     ("explain", node_explain), ("rank", node_rank),
                     ("aggregate", node_aggregate), ("retrieve", node_retrieve),
                     ("recommend", node_recommend), ("brief", node_brief),
                     ("rejected", node_rejected),
                     ("retrieve_context", retrieve_context),
                     ("synthesize", synthesize)]:
        g.add_node(name, fn)
    g.add_edge(START, "guardrail_in")
    g.add_edge("guardrail_in", "classify_intent")
    g.add_conditional_edges("classify_intent", route,
                            {"lookup": "lookup", "compare": "compare",
                             "forecast": "forecast", "scenario": "scenario",
                             "explain": "explain", "rank": "rank",
                             "aggregate": "aggregate", "retrieve": "retrieve",
                             "recommend": "recommend", "brief": "brief",
                             "rejected": "rejected"})
    for n in ("lookup", "compare", "forecast", "scenario", "explain",
              "rank", "aggregate", "retrieve", "recommend", "brief"):
        g.add_edge(n, "retrieve_context")
    g.add_edge("retrieve_context", "synthesize")
    g.add_edge("synthesize", END)
    g.add_edge("rejected", END)
    return g.compile()


_APP = None
def ask(question):
    global _APP
    if _APP is None:
        _APP = build_graph()
    out = _APP.invoke({"question": question})
    return out["answer"]


if __name__ == "__main__":
    for q in ["What is Maharashtra's conversion ratio?",
              "Compare Gujarat and Bihar",
              "What if subsidy increases by 10%?",
              "Why does the subsidy paradox matter?",
              "Predict installations for Mars"]:
        print("=" * 70, "\nQ:", q)
        print(ask(q).render())
