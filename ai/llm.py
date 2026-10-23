"""Answer generation for SuryaIQ.

`narrate()` turns a tool's STRUCTURED result into a proper natural-language
answer (deterministic NLG) — no raw JSON. If ANTHROPIC_API_KEY is set, Claude
polishes that grounded draft; otherwise the deterministic prose is returned
as-is, so the system gives clean answers with or without a key.
"""
import os
from pathlib import Path

# Load a local .env (gitignored) so the key never has to live in the shell or code.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

# Provider is auto-detected from whichever key is set. Override the model with
# SURYAIQ_MODEL in .env.
_OPENAI_DEFAULT = "gpt-4o-mini"
_OPENAI_FALLBACKS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]
_ANTHROPIC_DEFAULT = "claude-sonnet-5"
_ANTHROPIC_FALLBACKS = ["claude-sonnet-5", "claude-haiku-4-5-20251001", "claude-opus-5-5"]

_PLACEHOLDERS = ("sk-ant-replace", "sk-ant-your", "sk-ant-xxxx", "sk-ant-api03-xxxx",
                 "sk-your", "sk-xxxx", "sk-proj-xxxx", "sk-replace")


def _is_placeholder(k):
    return k.lower().startswith(_PLACEHOLDERS)


def provider():
    """Return 'openai', 'anthropic', or None based on the usable key present.
    OpenAI takes precedence when both are set."""
    ok = os.environ.get("OPENAI_API_KEY", "").strip()
    ak = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if ok and not _is_placeholder(ok) and ok.startswith("sk-") and len(ok) >= 20:
        return "openai"
    if ak and not _is_placeholder(ak) and ak.startswith("sk-ant-api") and len(ak) >= 90:
        return "anthropic"
    return None


def model_for(p=None):
    env = os.environ.get("SURYAIQ_MODEL", "").strip()
    if env:
        return env
    p = p or provider()
    return _OPENAI_DEFAULT if p == "openai" else _ANTHROPIC_DEFAULT


def key_status():
    """Return (usable, reason) without exposing the key."""
    ok = os.environ.get("OPENAI_API_KEY", "").strip()
    ak = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if ok:
        if _is_placeholder(ok):
            return False, "OpenAI key is placeholder/example text, not a real key"
        if not ok.startswith("sk-"):
            return False, "OpenAI key should start with 'sk-'"
        return True, "ok (openai)"
    if ak:
        if _is_placeholder(ak):
            return False, "Anthropic key is placeholder/example text, not a real key"
        if not ak.startswith("sk-ant-api"):
            return False, "Anthropic key should start with 'sk-ant-api03-'"
        if len(ak) < 90:
            return False, f"Anthropic key too short ({len(ak)} chars; ~108 expected)"
        return True, "ok (anthropic)"
    return False, "no key set (set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env)"


def have_key():
    return key_status()[0]


# back-compat for modules/tests that import MODEL
MODEL = model_for()


def _complete(system, user, max_tokens=600):
    p = provider()
    if p == "openai":
        return _complete_openai(system, user, max_tokens)
    if p == "anthropic":
        return _complete_anthropic(system, user, max_tokens)
    raise RuntimeError("no usable API key")


def _complete_openai(system, user, max_tokens):
    from openai import OpenAI, NotFoundError
    client = OpenAI()
    primary = model_for("openai")
    models = [primary] + [m for m in _OPENAI_FALLBACKS if m != primary]
    last_err = None
    for m in models:
        try:
            r = client.chat.completions.create(
                model=m, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return (r.choices[0].message.content or "").strip()
        except NotFoundError as e:
            last_err = e
            continue
    raise last_err


def _complete_anthropic(system, user, max_tokens):
    import anthropic
    client = anthropic.Anthropic()
    primary = model_for("anthropic")
    models = [primary] + [m for m in _ANTHROPIC_FALLBACKS if m != primary]
    last_err = None
    for m in models:
        try:
            msg = client.messages.create(model=m, max_tokens=max_tokens,
                                         system=system,
                                         messages=[{"role": "user", "content": user}])
            return "".join(b.text for b in msg.content
                           if getattr(b, "type", "") == "text").strip()
        except anthropic.NotFoundError as e:
            last_err = e
            continue
    raise last_err


def _pct(x):
    try:
        return f"{float(x)*100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(x):
    try:
        return f"{float(x):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


# ---------------------------------------------------------------- narration
def narrate(question, intent, tr, rag_hits=None):
    """Deterministic natural-language answer from a structured tool result."""
    tr = tr or {}
    rag_hits = rag_hits or []
    try:
        draft = _NARRATORS.get(intent, _n_default)(tr, rag_hits)
    except Exception:
        draft = _n_default(tr, rag_hits)
    if have_key():
        try:
            return _polish(question, intent, draft, tr, rag_hits)
        except Exception:
            return draft
    return draft


def _n_lookup(tr, rag):
    if "values" in tr:  # single state
        s, v = tr["state"], tr["values"]
        parts = [f"In **{s}**, {_num(v.get('installations'))} rooftop systems have "
                 f"been installed from {_num(v.get('applications_best'))} applications — "
                 f"an application-to-installation conversion ratio of "
                 f"{_pct(v.get('conversion_ratio'))}."]
        if v.get("subsidy_released_cr") is not None:
            parts.append(f"About ₹{_num(v.get('subsidy_released_cr'))} crore of central "
                         f"subsidy has been released.")
        if v.get("suitability_index") is not None:
            parts.append(f"Its rooftop-suitability score is "
                         f"{float(v['suitability_index']):.0f}/100.")
        return " ".join(parts)
    if "metric" in tr:  # single metric
        return (f"**{tr['state']}** — {tr['metric'].replace('_', ' ')}: "
                f"{tr.get('value')}.")
    return _n_aggregate(tr, rag)


def _n_rank(tr, rag):
    rk = tr.get("ranking", [])
    if not rk:
        return "No ranking could be computed for that metric."
    metric = tr["metric"].replace("_", " ")
    lead = rk[0]
    lead_val = _pct(lead["value"]) if "ratio" in tr["metric"] or "share" in tr["metric"] else _num(lead["value"])
    order = "highest" if tr["order"] == "descending" else "lowest"
    items = "; ".join(
        f"{i+1}. {r['state']} ({_pct(r['value']) if 'ratio' in tr['metric'] or 'share' in tr['metric'] else _num(r['value'])})"
        for i, r in enumerate(rk))
    return (f"**{lead['state']}** has the {order} {metric} ({lead_val}). "
            f"The {order} {len(rk)} states by {metric}: {items}.")


def _n_aggregate(tr, rag):
    parts = [f"Across **{tr.get('states')} States/UTs**, {_num(tr.get('total_installations'))} "
             f"installations have come from {_num(tr.get('total_applications'))} applications — "
             f"a national conversion ratio of {_pct(tr.get('national_conversion_ratio'))}."]
    if tr.get("total_subsidy_cr") is not None:
        parts.append(f"Total subsidy released is ₹{_num(tr.get('total_subsidy_cr'))} crore "
                     f"for {_num(tr.get('total_capacity_mw'))} MW of capacity.")
    for k, v in tr.items():
        if k.startswith("mean_"):
            parts.append(f"Mean {k[5:].replace('_',' ')} is {round(float(v),3)}.")
        if k.startswith("states_"):
            lab = k.replace("states_", "").replace("_", " ")
            ms = tr.get("matching_states", [])
            extra = f" ({', '.join(ms[:6])}{'…' if len(ms) > 6 else ''})" if ms else ""
            parts.append(f"**{v}** states are {lab}{extra}.")
    return " ".join(parts)


def _n_compare(tr, rag):
    metric = tr["metric"].replace("_", " ")
    pair = [(k, v) for k, v in tr.items() if k not in ("metric", "higher", "source")]
    if len(pair) < 2:
        return "Could not compare — need two valid states."
    (a, va), (b, vb) = pair[0], pair[1]
    fv = (lambda x: _pct(x)) if "ratio" in tr["metric"] or "share" in tr["metric"] else (lambda x: _num(x))
    return (f"**{tr['higher']}** has the higher {metric}. "
            f"{a}: {fv(va)} vs {b}: {fv(vb)}.")


def _n_forecast(tr, rag):
    if tr.get("scope") == "national":
        if tr.get("error"):
            return tr["error"]
        mape = tr.get("in_sample_mape_pct")
        mape_txt = f" (in-sample MAPE {mape:.1f}%)" if mape is not None else ""
        return (f"Nationally, PM Surya Ghar installations are projected to reach about "
                f"**{tr['point_M']}M** in {tr['horizon_months']} months "
                f"(range {tr['lo_M']}M–{tr['hi_M']}M), via log-linear extrapolation "
                f"from 6 dated national releases (Nov 2024–Aug 2026){mape_txt}. "
                f"Thin basis; the exponential assumption overstates the long horizon "
                f"if growth slows. This is an indicative trend projection, not a fitted monthly forecast.")
    if tr.get("target", "").startswith("PM Surya Ghar installations"):
        return (f"**{tr['state']}** currently has {_num(tr['current_installations'])} "
                f"installations. Applying its observed ~{_pct(tr['annual_growth_used'])} "
                f"annual growth, a {tr['horizon_months']}-month projection gives about "
                f"**{_num(tr['projected_installations'])}** (+{_num(tr['projected_additional'])}). "
                f"This is an indicative growth projection, not a fitted monthly forecast.")
    fc = tr.get("forecast", {})
    if not fc:
        return tr.get("error", "No forecast available.")
    last = fc["values"][-1]
    yr = fc["years"][-1]
    return (f"Using the **{tr.get('best_model')}** model, {tr['state']}'s RE installed "
            f"capacity is projected to reach about **{_num(last)} MW** by {yr} "
            f"(range {_num(fc['lower'][-1])}–{_num(fc['upper'][-1])} MW). "
            f"Predicted under model assumptions; {tr.get('note','')}")


def _n_scenario(tr, rag):
    if tr.get("supported") is False:
        return tr.get("note", "This scenario is not supported by the collected data.")
    p = tr["params"]
    lvl = []
    if p.get("subsidy_pct"):
        lvl.append(f"a {int(p['subsidy_pct']*100):+d}% subsidy change")
    if p.get("tariff_pct"):
        lvl.append(f"a {int(p['tariff_pct']*100):+d}% tariff change")
    lvl = " and ".join(lvl) or "the modelled intervention"
    # state-scoped answer when a specific state was named
    sd = tr.get("state_detail") or []
    if sd:
        parts = []
        for r in sd[:3]:
            parts.append(f"**{r['state']}**: conversion {_pct(r['conv_base'])} → "
                         f"{_pct(r['conv_scenario'])}, additional installations "
                         f"{_num(r['delta_installs'])}")
        return (f"Under {lvl} applied to {', '.join(r['state'] for r in sd)}: "
                + "; ".join(parts) + ". Predicted under model assumptions — "
                "associational, not causal.")
    base, scen = tr["national_installs_base"], tr["national_installs_scenario"]
    movers = tr.get("top_movers", [])[:3]
    mv = ", ".join(f"{m['state']} (+{_num(m['delta_installs'])})" for m in movers)
    return (f"Under {lvl}, national installations move from {_num(base)} to "
            f"**{_num(scen)}** ({tr['pct_change']:+.1f}%), with the subsidy burden at "
            f"₹{_num(tr['total_subsidy_burden_cr'])} crore. Biggest gains: {mv}. "
            f"Predicted under model assumptions — associational, not causal.")


def _n_explain(tr, rag):
    parts = []
    if tr.get("paradox"):
        p = tr["paradox"]
        if "linear_2feature_mae" in p:   # paper-faithful framing
            parts.append(
                f"The subsidy paradox: a 2-feature linear model (tariff growth + "
                f"irradiance) generalizes better (LOO-CV MAE {p['linear_2feature_mae']}) "
                f"than a 6-feature RandomForest ({p['random_forest_6feature_mae']}), "
                f"and subsidy/tariff level is among the weakest predictors. The "
                f"strongest correlates of conversion are "
                f"{' and '.join(p['strongest_predictors'])} — administrative and "
                f"physical-resource factors, not subsidy size (correlational, not causal).")
        else:
            parts.append(f"Implementation quality matters more than subsidy size: "
                         f"implementation-quality features predict conversion better "
                         f"(LOSO MAE {p['impl_quality_mae']}) than subsidy-amount features "
                         f"({p['subsidy_only_mae']}). Top drivers: "
                         f"{', '.join(f['feature'].replace('_',' ') for f in p['top_features'][:3])}.")
    for st, info in (tr.get("states") or {}).items():
        v = info.get("values", {})
        if v:
            parts.append(f"{st}: conversion {_pct(v.get('conversion_ratio'))}, "
                         f"{_num(v.get('installations'))} installs, suitability "
                         f"{float(v['suitability_index']):.0f}/100." if v.get("suitability_index") is not None
                         else f"{st}: conversion {_pct(v.get('conversion_ratio'))}.")
    if rag:
        parts.append("Sources: " + _cite(rag))
    return " ".join(parts) or _n_default(tr, rag)


def _n_retrieve(tr, rag):
    passages = tr.get("passages", [])
    if not passages:
        return "No matching passages were found in the project's sources."
    bullets = "\n".join(f"- *({p['source']})* {p['text'].strip()[:200]}…"
                        for p in passages[:3])
    return ("Here is what the project's own sources say:\n" + bullets)


def _n_recommend(tr, rag):
    recs = tr.get("priority_states", [])
    names = ", ".join(f"**{r['state']}**" for r in recs)
    detail = "; ".join(f"{r['state']} (conversion {_pct(r['conversion'])}, "
                       f"potential {r['solar_potential_gw']} GW)" for r in recs[:5])
    return (f"{tr['recommendation']} Priority states: {names}. "
            f"These combine high solar potential with below-median conversion — {detail}.")


def _n_default(tr, rag):
    if isinstance(tr, dict) and tr.get("error"):
        return f"I couldn't answer that: {tr['error']}."
    if rag:
        return "From the project's sources: " + _cite(rag)
    return "I don't have enough grounded data to answer that precisely."


def _cite(rag):
    return "; ".join(f"{h['source']}" for h in rag[:3])


def _n_brief(tr, rag):
    if tr.get("error"):
        return tr["error"]
    s = tr["snapshot"]
    lines = [f"### Executive brief — **{tr['state']}**", ""]
    lines.append(f"**Snapshot.** Conversion {_pct(s['conversion_ratio'])} "
                 f"({s.get('rank_by_conversion','?')} by rank); "
                 f"{_num(s['installations'])} installations from "
                 f"{_num(s['applications'])} applications; "
                 f"₹{_num(s['subsidy_released_cr'])} cr subsidy released; "
                 f"suitability {s.get('suitability_index','?')}/100.")
    if tr.get("top_drivers"):
        lines.append("")
        lines.append(f"**Top drivers (SHAP).** " + "; ".join(tr["top_drivers"]) + ".")
    if tr.get("scenario_D_tariff_plus10"):
        lines.append("")
        lines.append(f"**Scenario D (tariff +10%).** " + tr["scenario_D_tariff_plus10"])
    if tr.get("paper_finding"):
        lines.append("")
        lines.append(f"**Paper evidence.** {tr['paper_finding']}")
    lines.append("")
    lines.append(f"**Recommendation.** {tr['recommendation']}")
    return "\n".join(lines)


_NARRATORS = {"lookup": _n_lookup, "rank": _n_rank, "aggregate": _n_aggregate,
              "compare": _n_compare, "forecast": _n_forecast,
              "scenario": _n_scenario, "explain": _n_explain,
              "retrieve": _n_retrieve, "recommend": _n_recommend,
              "brief": _n_brief}


def _polish(question, intent, draft, tr, rag):
    import json
    system = (
        "You are SuryaIQ, a policy analyst for India's PM Surya Ghar rooftop-solar "
        "scheme. Rewrite the DRAFT into a clear, concise 2-4 sentence answer to the "
        "user's question. Use ONLY the numbers in the draft and evidence — never "
        "invent figures. Keep every number and any 'predicted under model "
        "assumptions' / 'not causal' caveat. Plain, professional tone.")
    ev = json.dumps(tr, default=str)[:1200]
    user = (f"Question: {question}\nIntent: {intent}\nDRAFT: {draft}\n"
            f"Structured evidence: {ev}\nSources: {_cite(rag)}")
    return _complete(system, user)
