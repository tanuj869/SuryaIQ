"""LangChain-style typed tools. Each returns structured data + the source
file(s) used, so every downstream answer is attributable. Pure/deterministic;
no LLM required."""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ai"))

import models        # noqa: E402
import scenarios     # noqa: E402
from retriever import get_retriever  # noqa: E402

PANEL = ROOT / "data" / "processed" / "master_state_panel.csv"
_METRIC_ALIASES = {
    "conversion": "conversion_ratio", "conversion ratio": "conversion_ratio",
    "installations": "installations", "installs": "installations",
    "applications": "applications_best", "subsidy": "subsidy_released_cr",
    "capacity": "capacity_mw", "tariff": "tariff_2kw_2025",
    "potential": "solar_potential_gw", "irradiance": "annual_ghi",
    "suitability": "suitability_index", "adoption gap": "adoption_gap",
}


def _panel():
    return pd.read_csv(PANEL)


METRIC_WORDS = {
    "installation": "installations", "install": "installations",
    "conversion": "conversion_ratio", "application": "applications_best",
    "registration": "registrations", "subsidy": "subsidy_released_cr",
    "capacity": "capacity_mw", "tariff": "tariff_2kw_2025",
    "potential": "solar_potential_gw", "irradiance": "annual_ghi",
    "ghi": "annual_ghi", "sun": "annual_ghi", "suitability": "suitability_index",
    "household": "households_benefitted", "gap": "adoption_gap",
    "renewable": "re_share_2025",
}


def detect_metric(text, default="installations"):
    t = str(text).lower()
    # "adoption" alone = volume (installations) — matches the report's "highest
    # adoption -> Gujarat"; "adoption rate/ratio/conversion" = conversion_ratio
    if "adoption" in t or "adopt" in t:
        return "conversion_ratio" if re.search(r"rate|ratio|conversion|percent", t) else "installations"
    for w, col in METRIC_WORDS.items():
        if w in t:
            return col
    return default


def rank_states(metric="installations", n=5, ascending=False):
    col = _METRIC_ALIASES.get(str(metric).lower(), metric)
    d = (_panel().dropna(subset=[col])[["state", col]]
         .sort_values(col, ascending=ascending).head(n))
    return {"metric": col, "order": "ascending" if ascending else "descending",
            "ranking": [{"state": r["state"], "value": round(float(r[col]), 4)}
                        for _, r in d.iterrows()],
            "source": "master_state_panel.csv"}


def national_summary(query=None):
    d = _panel()
    apps, inst = d["applications_best"].sum(), d["installations"].sum()
    out = {"states": int(d["state"].nunique()),
           "total_applications": int(apps), "total_installations": int(inst),
           "national_conversion_ratio": round(inst / apps, 4),
           "total_subsidy_cr": round(d["subsidy_released_cr"].sum(), 1),
           "total_capacity_mw": round(d["capacity_mw"].sum(), 1),
           "source": "master_state_panel.csv"}
    if query:
        col = detect_metric(query, default=None)
        if col and col in d:
            out[f"mean_{col}"] = round(float(d[col].mean()), 4)
            out[f"median_{col}"] = round(float(d[col].median()), 4)
        import re as _re
        m = _re.search(r"(above|over|below|under|greater than|less than)\s+([\d.]+)%?",
                       str(query).lower())
        if m and col and col in d:
            thr = float(m.group(2))
            thr = thr / 100 if col == "conversion_ratio" and thr > 1 else thr
            cmp = d[col] > thr if m.group(1) in ("above", "over", "greater than") else d[col] < thr
            out[f"states_{m.group(1)}_{m.group(2)}"] = int(cmp.sum())
            out["matching_states"] = d[cmp]["state"].tolist()
    return out


def _find_state(name):
    d = _panel()
    name_l = str(name).lower().strip()
    for s in d["state"]:
        if name_l == s.lower() or name_l in s.lower() or s.lower() in name_l:
            return s
    return None


def query_state(state, metric=None):
    s = _find_state(state)
    if not s:
        return {"error": f"state '{state}' not found", "source": "master_state_panel.csv"}
    row = _panel().set_index("state").loc[s]
    if metric:
        col = _METRIC_ALIASES.get(metric.lower(), metric)
        if col not in row.index:
            return {"error": f"metric '{metric}' unknown", "source": "master_state_panel.csv"}
        return {"state": s, "metric": col, "value": _clean(row[col]),
                "source": "master_state_panel.csv"}
    keep = ["applications_best", "installations", "conversion_ratio",
            "subsidy_released_cr", "suitability_index", "adoption_gap"]
    return {"state": s, "values": {k: _clean(row[k]) for k in keep},
            "source": "master_state_panel.csv"}


def compare_states(a, b, metric="conversion_ratio"):
    col = _METRIC_ALIASES.get(str(metric).lower(), metric)
    ra, rb = query_state(a, col), query_state(b, col)
    if "error" in ra or "error" in rb:
        return {"error": ra.get("error") or rb.get("error")}
    va, vb = ra["value"], rb["value"]
    lead = ra["state"] if (va or 0) >= (vb or 0) else rb["state"]
    return {"metric": col, ra["state"]: va, rb["state"]: vb,
            "higher": lead, "source": "master_state_panel.csv"}


def run_forecast(state, horizon=3):
    s = _find_state(state) or state
    return {**models.forecast_re_capacity(s, horizon),
            "source": "re_capacity_panel.csv (MNRE File 10.2)"}


def run_scenario(name="B_subsidy_+10%", subsidy_pct=0.10, tariff_pct=0.0,
                 target_states=None):
    # Paper honesty: a subsidy scenario is NOT supported by the collected data
    # (no state-level subsidy top-up feature was compiled). Flag it.
    if subsidy_pct and not tariff_pct:
        return {"scenario": name, "supported": False,
                "note": "Scenario B (subsidy change) is NOT supportable by the "
                        "published model: state-level subsidy top-up data was "
                        "deliberately scoped out (no centralized source), so the "
                        "model has no subsidy feature to perturb. Supportable "
                        "levers are tariff-growth and high-potential targeting "
                        "(Scenarios C/D). This mirrors the paper's scenario-support "
                        "assessment (Table IV).",
                "source": "paper_results.json (Table IV)"}
    summary, detail = scenarios.scenario(name=name, subsidy_pct=subsidy_pct,
                                         tariff_pct=tariff_pct,
                                         target_states=target_states)
    summary["source"] = "master_state_panel.csv + RandomForest conversion model"
    summary["top_movers"] = (detail.sort_values("delta_installs", ascending=False)
                             .head(5)[["state", "delta_installs"]]
                             .to_dict("records"))
    # when scoped to specific states, surface their own before/after numbers
    if target_states:
        scoped = detail[detail["state"].isin(target_states)]
        summary["target_states"] = target_states
        summary["state_detail"] = [
            {"state": r["state"],
             "conv_base": round(float(r["conv_pred_base"]), 3),
             "conv_scenario": round(float(r["conv_pred_scenario"]), 3),
             "delta_installs": int(r["delta_installs"])}
            for _, r in scoped.iterrows()]
    return summary


def subsidy_paradox():
    """Report the published (paper-faithful) subsidy-paradox finding:
    a 2-feature linear model (tariff growth + irradiance) beats a 6-feature
    RandomForest under LOO-CV, and subsidy/tariff LEVEL is among the weakest
    predictors."""
    import json
    f = ROOT / "data" / "processed" / "paper_results.json"
    if f.exists():
        r = json.loads(f.read_text())
        mc = r["table2_model_comparison"]
        corr = r["table1_correlations"]
        strongest = [c["feature"] for c in corr[:2]]
        weakest = [c["feature"] for c in corr if "level" in c["feature"].lower()
                   or "Subsidy" in c["feature"]][:2]
        return {"verdict": mc["verdict"],
                "linear_2feature_mae": mc["linear_2feature"]["loo_cv_mae"],
                "random_forest_6feature_mae": mc["random_forest_6feature"]["loo_cv_mae"],
                "winner": mc["winner"],
                "strongest_predictors": strongest,
                "weakest_predictors": weakest or ["Tariff level 2024", "Tariff level 2025"],
                "source": "paper_results.json (LOO-CV; Gore & Sonawane)"}
    # fallback to the older RF feature-set comparison
    cached = ROOT / "data" / "processed" / "subsidy_paradox_results.json"
    r = json.loads(cached.read_text()) if cached.exists() else models.subsidy_paradox_test()
    return {"verdict": r["verdict"],
            "subsidy_only_mae": r["subsidy_amount_only"]["loso_mae"],
            "impl_quality_mae": r["implementation_quality_only"]["loso_mae"],
            "top_features": r["feature_importance_full"][:4],
            "source": "subsidy_paradox_results.json (LOSO RandomForest)"}


def run_forecast_installations(state, months=12):
    s = _find_state(state) or state
    return {**models.forecast_installations(s, months),
            "source": "master_state_panel.csv (files 2/4 snapshot + 10.1 growth)"}


def national_forecast(months=12):
    """National installation forecast via the paper's log-linear extrapolation
    (Table III). Routed when the user asks for 'installations in N months' and
    does NOT name a specific state."""
    import json as _json
    import math
    import numpy as _np
    paper = _json.loads((ROOT / "data/processed/paper_results.json").read_text())
    tab3 = paper.get("table3_national_forecast", {})
    # try to return exactly the horizon the user asked for; otherwise interpolate
    hs = {h["horizon_months"]: h for h in tab3.get("horizons", [])}
    if months in hs:
        hz = hs[months]
    else:
        # linear-in-log interpolation between the two nearest published horizons
        xs = sorted(hs)
        if not xs:
            return {"error": "national forecast not computed; run src/paper_analysis.py"}
        lo = max([x for x in xs if x <= months], default=xs[0])
        hi = min([x for x in xs if x >= months], default=xs[-1])
        if lo == hi:
            hz = hs[lo]
        else:
            a, b = hs[lo], hs[hi]
            t = (months - lo) / (hi - lo)
            def _mix(k):
                return float(_np.exp(math.log(a[k]) * (1 - t) + math.log(b[k]) * t))
            hz = {"horizon_months": months, "point": round(_mix("point"), 2),
                  "lo_M": round(_mix("lo_M"), 2), "hi_M": round(_mix("hi_M"), 2)}
    return {"scope": "national", "horizon_months": months,
            "point_M": hz["point"], "lo_M": hz["lo_M"], "hi_M": hz["hi_M"],
            "method": "log-linear extrapolation from 6 dated national releases "
                      "(Nov 2024–Aug 2026)",
            "in_sample_mape_pct": (tab3.get("in_sample_mape") or {}).get("log_linear"),
            "note": tab3.get("note"),
            "source": "paper_results.json (Table III)"}


def recommend(n=5):
    """Policy recommendation: which states to prioritize. Targets states with
    high solar potential / suitability but weak conversion (adoption gap)."""
    d = _panel()
    med_pot = d["solar_potential_gw"].median()
    med_conv = d["conversion_ratio"].median()
    cand = d[(d["solar_potential_gw"] > med_pot) & (d["conversion_ratio"] < med_conv)]
    cand = cand.sort_values("adoption_gap").head(n)
    recs = [{"state": r["state"],
             "conversion": round(float(r["conversion_ratio"]), 3),
             "solar_potential_gw": round(float(r["solar_potential_gw"]), 1),
             "adoption_gap": round(float(r["adoption_gap"]), 1) if pd.notna(r["adoption_gap"]) else None}
            for _, r in cand.iterrows()]
    return {"recommendation": "Prioritize high-potential, low-conversion states "
            "for differentiated incentives / DISCOM support — focus on "
            "implementation quality (disbursement speed, conversion), which the "
            "model finds matters more than subsidy size.",
            "priority_states": recs,
            "source": "master_state_panel.csv + subsidy_paradox_results.json"}


def summarize(query, k=4):
    """Retrieval-answer: pull the most relevant passages from the corpus
    (reports/CSVs/literature) for 'summarize' / 'show evidence' questions."""
    hits = get_retriever().search(query, k=k)
    return {"query": query,
            "passages": [{"source": h["source"], "text": h["text"][:280]} for h in hits],
            "sources": [h["source"] for h in hits]}


def state_brief(state):
    """Generate a per-state executive policy brief — the kind an energy
    secretariat would want on their desk. Combines data, SHAP decomposition,
    and the paper's model to produce an actionable, cited one-pager."""
    import json
    st = _find_state(state) or state
    data = query_state(st)
    if "error" in data:
        return data
    v = data["values"]
    # SHAP contributions
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "src"))
        from explain import shap_for_state
        shap_data = shap_for_state(st)
    except Exception:
        shap_data = {}
    # tariff scenario predicted for THIS state
    tariff_scen = run_scenario(name="brief_tariff", subsidy_pct=0.0,
                               tariff_pct=0.10, target_states=[st])
    # cite paper verdict
    paradox = subsidy_paradox()
    # ranking position
    d = _panel().dropna(subset=["conversion_ratio"])
    rank = int(d["conversion_ratio"].rank(ascending=False).loc[d["state"] == st].iloc[0]) \
           if (d["state"] == st).any() else None
    total = int(d["state"].nunique())
    # drivers (top |SHAP|)
    drivers = []
    for c in (shap_data.get("contributions") or [])[:3]:
        direction = "lifts" if c["shap"] > 0 else "drags"
        drivers.append(f"{c['feature'].replace('_', ' ')} {direction} the "
                       f"prediction by {abs(c['shap']):.3f} (value {c['value']})")
    # scenario answer
    scen_line = ""
    if tariff_scen.get("state_detail"):
        r = tariff_scen["state_detail"][0]
        scen_line = (f"A +10% tariff-growth shock lifts predicted conversion "
                     f"from {r['conv_base']:.3f} to {r['conv_scenario']:.3f} "
                     f"(additional ~{r['delta_installs']:,} installations under model assumptions).")
    return {
        "state": st,
        "snapshot": {
            "conversion_ratio": v.get("conversion_ratio"),
            "installations": v.get("installations"),
            "applications": v.get("applications_best"),
            "subsidy_released_cr": v.get("subsidy_released_cr"),
            "suitability_index": v.get("suitability_index"),
            "rank_by_conversion": f"{rank}/{total}" if rank else None,
        },
        "paper_finding": paradox.get("verdict"),
        "top_drivers": drivers,
        "scenario_D_tariff_plus10": scen_line,
        "recommendation": _state_recommendation(v, shap_data),
        "sources": ["master_state_panel.csv", "shap_summary.json",
                    "paper_results.json"],
        "format": "executive_brief_v1",
    }


def _state_recommendation(v, shap_data):
    conv = v.get("conversion_ratio") or 0
    if conv >= 0.70:
        return ("High-performing state: maintain execution and share best practices "
                "with lagging DISCOMs; marginal subsidy increases are unlikely to lift "
                "already-near-ceiling conversion.")
    if conv >= 0.40:
        return ("Mid-performing state: target the application-to-installation pipeline "
                "(portal, inspection and net-metering turnaround) rather than headline "
                "subsidy; tariff-growth dynamics are the strongest observable lever.")
    return ("Lagging state: diagnose administrative friction first (disbursement time, "
            "DISCOM load, awareness) before adding subsidy. The paper's cross-state "
            "evidence shows subsidy level is a weak explanatory variable; execution is.")


def retrieve_docs(query, k=4):
    hits = get_retriever().search(query, k=k)
    return {"query": query, "hits": hits,
            "sources": [h["source"] for h in hits]}


def _clean(v):
    try:
        f = float(v)
        return round(f, 4)
    except (TypeError, ValueError):
        return None if pd.isna(v) else v


if __name__ == "__main__":
    import json
    print(json.dumps(query_state("Maharashtra"), indent=2))
    print(json.dumps(compare_states("Gujarat", "Bihar"), indent=2))
    print(json.dumps(run_scenario(), indent=2)[:500])
    print(json.dumps(subsidy_paradox(), indent=2))
