"""SuryaIQ dashboard — PM Surya Ghar rooftop-solar adoption & policy assistant.

Run:  streamlit run app/streamlit_app.py
Offline-safe. The AI tab uses Claude when ANTHROPIC_API_KEY is set, else the
deterministic offline assistant.
"""
import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ai"))
PROC = ROOT / "data" / "processed"

st.set_page_config(page_title="SuryaIQ", page_icon="☀️", layout="wide")
import map_helper  # noqa: E402
PRIMARY = "#F2A900"
SEQ = "Oranges"


@st.cache_data
def panel():
    return pd.read_csv(PROC / "master_state_panel.csv")


@st.cache_data
def re_panel():
    return pd.read_csv(PROC / "re_capacity_panel.csv")


@st.cache_data
def paradox():
    f = PROC / "subsidy_paradox_results.json"
    return json.loads(f.read_text()) if f.exists() else None


def kpi_row(df):
    apps = df["applications_best"].sum()
    inst = df["installations"].sum()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("States / UTs", f"{df['state'].nunique()}")
    c2.metric("Applications", f"{apps/1e6:.2f} M")
    c3.metric("Installations", f"{inst/1e6:.2f} M")
    c4.metric("National conversion", f"{inst/apps*100:.1f}%")
    c5.metric("Subsidy released", f"₹{df['subsidy_released_cr'].sum():,.0f} cr")


# ----------------------------------------------------------------- views
def view_overview(df):
    st.subheader("National overview")
    kpi_row(df)
    # India choropleth — conversion ratio across states.
    # Fixed figure width + no stretch avoids the plotly geo re-fit shrink bug.
    _fig = map_helper.choropleth(
        df.dropna(subset=["conversion_ratio"]),
        "conversion_ratio",
        "Application → installation conversion ratio (23-Aug-2026 snapshot)",
        colorbar_title="conversion")
    st.plotly_chart(_fig, use_container_width=False,
                    config={"displayModeBar": False})
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Conversion by state (ranked)**")
        d = df.dropna(subset=["conversion_ratio"]).sort_values("conversion_ratio")
        fig = px.bar(d, x="conversion_ratio", y="state", orientation="h",
                     color="conversion_ratio", color_continuous_scale=SEQ,
                     height=720, labels={"conversion_ratio": "conversion ratio"})
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Installations (log scale)**")
        d = df.dropna(subset=["installations"]).sort_values("installations")
        fig = px.bar(d.tail(20), x="installations", y="state", orientation="h",
                     color_discrete_sequence=[PRIMARY], height=720, log_x=True)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)


def view_state(df):
    st.subheader("State explorer")
    s = st.selectbox("State / UT", sorted(df["state"]), index=sorted(df["state"]).index("Maharashtra")
                     if "Maharashtra" in list(df["state"]) else 0)
    r = df.set_index("state").loc[s]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conversion", f"{r['conversion_ratio']*100:.1f}%" if pd.notna(r["conversion_ratio"]) else "—")
    c2.metric("Installations", f"{r['installations']:,.0f}" if pd.notna(r["installations"]) else "—")
    c3.metric("Suitability", f"{r['suitability_index']:.0f}/100" if pd.notna(r["suitability_index"]) else "—")
    c4.metric("Subsidy", f"₹{r['subsidy_released_cr']:,.0f} cr" if pd.notna(r["subsidy_released_cr"]) else "—")

    st.markdown("**Adoption funnel**")
    stages = ["registrations", "applications_best", "installations", "households_benefitted"]
    labels = ["Registrations", "Applications", "Installations", "Households benefitted"]
    vals = [r.get(c) for c in stages]
    fig = go.Figure(go.Funnel(y=labels, x=vals, marker=dict(color=PRIMARY),
                              textinfo="value+percent initial"))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**RE-capacity trajectory (MNRE, FY2017-18 → 2024-25)**")
    rp = re_panel()
    sp = rp[rp["state"] == s]
    if len(sp):
        fig = px.line(sp, x="fy", y="re_capacity_mw", markers=True,
                      color_discrete_sequence=[PRIMARY], labels={"re_capacity_mw": "RE capacity (MW)", "fy": ""})
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)


def _label_only_outliers(d, metric, k=8):
    """Only label the k biggest states (by metric) so dot-labels don't overlap."""
    top = set(d.nlargest(k, metric)["state"])
    return d["state"].where(d["state"].isin(top), "")


def view_drivers(df):
    st.subheader("What drives adoption?")
    st.caption("Hover any dot for the state name — labels are shown only for the "
               "largest 8 states to avoid overlap.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Solar potential vs installed capacity** (adoption gap)")
        d = df.dropna(subset=["solar_potential_gw", "capacity_mw"]).copy()
        d["label"] = _label_only_outliers(d, "capacity_mw", k=8)
        fig = px.scatter(d, x="solar_potential_gw", y="capacity_mw", text="label",
                         size="applications_best", color="conversion_ratio",
                         hover_name="state",
                         color_continuous_scale=SEQ, log_x=True, log_y=True, height=480)
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Residential tariff vs conversion**")
        d = df.dropna(subset=["tariff_2kw_2025", "conversion_ratio"]).copy()
        d["label"] = _label_only_outliers(d, "conversion_ratio", k=8)
        fig = px.scatter(d, x="tariff_2kw_2025", y="conversion_ratio", text="label",
                         color="suitability_index", hover_name="state",
                         color_continuous_scale=SEQ, height=480,
                         labels={"tariff_2kw_2025": "tariff ₹/kWh", "conversion_ratio": "conversion"})
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    par = paradox()
    if par:
        st.markdown("**Subsidy-paradox test** — leave-one-state-out RandomForest MAE (lower = better)")
        rows = [("Subsidy-amount only", par["subsidy_amount_only"]["loso_mae"]),
                ("Implementation-quality only", par["implementation_quality_only"]["loso_mae"]),
                ("Full (subsidy+impl+context)", par["subsidy+impl+context (full)"]["loso_mae"])]
        pdf = pd.DataFrame(rows, columns=["feature set", "LOSO MAE"])
        fig = px.bar(pdf, x="LOSO MAE", y="feature set", orientation="h",
                     color_discrete_sequence=[PRIMARY], height=220)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.info(par["verdict"])
        imp = pd.DataFrame(par["feature_importance_full"])
        st.markdown("**Feature importance (full model)**")
        fig = px.bar(imp.sort_values("importance"), x="importance", y="feature",
                     orientation="h", color_discrete_sequence=[PRIMARY], height=320)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)


def view_scenarios(df):
    st.subheader("Policy scenario simulator")
    st.caption("Associational, not causal — results are *predicted under model "
               "assumptions*. The simulator exposes only the levers the paper's "
               "model supports: tariff-growth shocks (linear coefficient) and "
               "high-potential-state targeting (Scenarios C/D, Table IV). Subsidy "
               "level is not a lever — it is a weak predictor with no centralized "
               "data source (Scenario B unsupportable); solar irradiance is a strong "
               "predictor but fixed by geography, so neither is policy-actionable here.")
    import scenarios as SC
    subsidy = 0.0
    tariff = st.slider("Tariff-growth shock (%)", -20, 30, 10, 5) / 100
    target_only = st.checkbox("Apply only to high-potential, low-adoption states", value=False)
    pan = df
    med_pot, med_conv = pan["solar_potential_gw"].median(), pan["conversion_ratio"].median()
    targets = pan[(pan["solar_potential_gw"] > med_pot) &
                  (pan["conversion_ratio"] < med_conv)]["state"].tolist()
    summary, detail = SC.scenario(name="interactive", subsidy_pct=subsidy, tariff_pct=tariff,
                                  target_states=targets if target_only else None)
    c1, c2, c3 = st.columns(3)
    c1.metric("Installs (base)", f"{summary['national_installs_base']:,}")
    c2.metric("Installs (scenario)", f"{summary['national_installs_scenario']:,}",
              f"{summary['pct_change']:+.1f}%")
    c3.metric("Subsidy burden", f"₹{summary['total_subsidy_burden_cr']:,.0f} cr")
    if target_only:
        st.caption("Targeted states: " + ", ".join(targets))
    # sort ascending so largest gain is at the top of a horizontal bar chart
    top = detail.sort_values("delta_installs", ascending=False).head(12).sort_values("delta_installs")
    fig = px.bar(top, x="delta_installs", y="state", orientation="h",
                 color_discrete_sequence=[PRIMARY], height=420,
                 labels={"delta_installs": "additional installations"})
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                      yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)


def view_models(df):
    st.subheader("Models & training")
    st.caption("Every dataset trains into the model set (ARIMA/SARIMAX + "
               "gradient-boosted / RandomForest), per the AI Summary Report.")
    f = PROC / "training_report.json"
    if not f.exists():
        st.warning("Run `python src/train.py` to generate the training report.")
        return
    rep = json.loads(f.read_text())
    st.markdown("**Dataset → model usage**")
    st.dataframe(pd.DataFrame(list(rep["dataset_usage"].items()),
                 columns=["dataset", "how it trains the model"]),
                 use_container_width=True, hide_index=True)
    cm = rep["models"]["conversion_regression"]["model_comparison_LOSO"]
    rows = [{"model": k, "LOSO MAE": v["loso_mae"], "LOSO R²": v["loso_r2"]}
            for k, v in cm.items() if isinstance(v, dict)]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Conversion model comparison (lower MAE = better)**")
        mdf = pd.DataFrame(rows).sort_values("LOSO MAE")
        fig = px.bar(mdf, x="LOSO MAE", y="model", orientation="h",
                     color_discrete_sequence=[PRIMARY], height=240)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Best: **{cm['best_model']}**")
    with c2:
        st.markdown("**Top-quartile adopter classifier**")
        cls = rep["models"]["topquartile_classification"]
        st.metric("LOSO accuracy", f"{cls['loso_accuracy']*100:.0f}%")
        st.metric("LOSO F1", f"{cls['loso_f1']:.2f}")
    st.markdown("**Time-series forecast — best model per state** (RE-capacity panel)")
    ts = rep["models"]["re_capacity_forecast"]["per_state"]
    st.dataframe(pd.DataFrame([{"state": k, "best model": v["best_model"],
                 **{f"MAE·{mk}": mv for mk, mv in v["backtest_mae"].items()}}
                 for k, v in ts.items()]), use_container_width=True, hide_index=True)


API_URL = os.environ.get("SURYAIQ_API", "http://localhost:8000")


def _ask_via_api(question):
    """Call the FastAPI /ask endpoint (Dashboard -> FastAPI -> LangGraph).
    Returns (answer_dict, 'api') or (None, reason) on failure."""
    try:
        import httpx
        r = httpx.post(f"{API_URL}/ask", json={"question": question}, timeout=30)
        r.raise_for_status()
        return r.json(), "api"
    except Exception as e:
        return None, str(e)


def view_explainability(df):
    st.subheader("Model explainability (SHAP)")
    st.caption("Per-state SHAP decomposition of the paper's 6-feature RandomForest "
               "conversion model. SHAP values show how each feature pushes the "
               "prediction above or below the model's base rate.")
    f = PROC / "shap_summary.json"
    if not f.exists():
        st.warning("Run `python src/explain.py` to compute SHAP values.")
        return
    sh = json.loads(f.read_text())
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Global feature importance (mean |SHAP|)**")
        g = pd.DataFrame(sorted(sh["global_importance"].items(),
                                key=lambda x: x[1]),
                         columns=["feature", "mean_abs_shap"])
        fig = px.bar(g, x="mean_abs_shap", y="feature", orientation="h",
                     color_discrete_sequence=[PRIMARY], height=320)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.info("Irradiance dominates globally — consistent with the paper's "
                "finding that physical-resource endowment matters more than "
                "subsidy size.")
    with c2:
        states = sorted(sh["by_state"].keys())
        state = st.selectbox("State", states, index=states.index("Maharashtra")
                             if "Maharashtra" in states else 0, key="shap_state")
        info = sh["by_state"][state]
        st.metric("Base conversion (model mean)", f"{sh['base_value']:.3f}")
        d1, d2, d3 = st.columns(3)
        d1.metric("Predicted", f"{info['prediction']:.3f}")
        d2.metric("Actual", f"{info['actual']:.3f}")
        d3.metric("Δ from base", f"{info['prediction']-sh['base_value']:+.3f}")
        rows = sorted([{"feature": k, "shap": v, "value": info["values"][k]}
                       for k, v in info["shap"].items()],
                      key=lambda r: r["shap"])
        bar = go.Figure(go.Bar(
            x=[r["shap"] for r in rows],
            y=[r["feature"] for r in rows], orientation="h",
            marker_color=["#A2D7A2" if r["shap"] >= 0 else "#D77A7A" for r in rows],
            text=[f"{r['shap']:+.3f} (val {r['value']})" for r in rows],
            textposition="auto",
        ))
        bar.update_layout(title=f"SHAP contributions for {state}",
                          margin=dict(l=0, r=0, t=40, b=0), height=320)
        st.plotly_chart(bar, use_container_width=True)


def view_brief(df):
    st.subheader("One-click executive brief")
    st.caption("Generates a cited, SHAP-explained, scenario-tested policy brief "
               "for any state — the kind of memo an energy secretariat would want on "
               "its desk.")
    states = sorted(df["state"].unique())
    state = st.selectbox("State / UT", states,
                         index=states.index("Maharashtra") if "Maharashtra" in states else 0,
                         key="brief_state")
    if st.button(f"📝 Generate brief for {state}", use_container_width=True):
        import graph
        with st.spinner("Composing…"):
            a = graph.ask(f"Give me an executive brief for {state}")
        st.markdown(a.answer)
        st.caption("Sources: " + ", ".join(a.data_source))


def view_assistant(df):
    st.subheader("Ask SuryaIQ")
    import llm
    import graph
    prov = llm.provider()
    mode = f"{prov} ({llm.model_for(prov)})" if prov else "offline (deterministic, cited)"
    # is the FastAPI orchestration layer reachable?
    api_up = False
    try:
        import httpx
        api_up = httpx.get(f"{API_URL}/health", timeout=2).status_code == 200
    except Exception:
        api_up = False
    route = "FastAPI → LangGraph" if api_up else "in-process LangGraph"
    st.caption(f"Mode: {mode} · Routed via **{route}**. "
               f"Guardrailed — out-of-scope questions are declined.")
    examples = ["Which state had the highest adoption?",
                "Forecast installations for the next 12 months",
                "Why does the subsidy paradox matter?",
                "Generate a policy recommendation"]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, use_container_width=True):
            st.session_state["suryaiq_q"] = ex
    q = st.text_input("Your question", key="suryaiq_q",
                      placeholder="Ask about conversion, forecasts, scenarios…")
    if q:
        with st.spinner("Thinking…"):
            data, how = (_ask_via_api(q) if api_up else (None, "in-process"))
            if data is None:          # API down or errored -> in-process fallback
                a = graph.ask(q)
                data = {"intent": a.intent, "confidence": a.confidence,
                        "answer": a.answer, "forecast": a.forecast,
                        "data_source": a.data_source, "assumptions": a.assumptions,
                        "limitations": a.limitations}
        st.markdown(f"**Intent:** `{data['intent']}` · **Confidence:** `{data['confidence']}`")
        st.markdown(data["answer"])
        if data.get("forecast") and data["intent"] in ("forecast", "scenario"):
            with st.expander("Model output"):
                st.json(data["forecast"])
        st.markdown("**Sources:** " + ", ".join(data["data_source"]))
        if data.get("assumptions"):
            st.caption("Assumptions: " + "; ".join(data["assumptions"]))
        if data.get("limitations"):
            st.caption("Limitations: " + "; ".join(data["limitations"]))


# ----------------------------------------------------------------- shell
def main():
    df = panel()
    st.sidebar.title("☀️ SuryaIQ")
    st.sidebar.caption("PM Surya Ghar adoption — forecasting & policy decision support")
    view = st.sidebar.radio("Navigate", ["Overview", "State explorer",
                            "Drivers & paradox", "Explainability (SHAP)",
                            "Scenario simulator", "Executive brief",
                            "Models & training", "Ask SuryaIQ"])
    st.sidebar.divider()
    st.sidebar.caption(f"{df['state'].nunique()} States/UTs · "
                       f"{df['installations'].sum()/1e6:.2f}M installs")
    {"Overview": view_overview, "State explorer": view_state,
     "Drivers & paradox": view_drivers,
     "Explainability (SHAP)": view_explainability,
     "Scenario simulator": view_scenarios,
     "Executive brief": view_brief,
     "Models & training": view_models,
     "Ask SuryaIQ": view_assistant}[view](df)
    st.divider()
    st.caption("Data: PM Surya Ghar portal, MNRE, NITI ICeD, NASA POWER, SERC tariff orders. "
               "Snapshots differ by date (Nov 2024–Aug 2026). Scenarios are associational, not causal.")


main()
