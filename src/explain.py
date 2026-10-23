"""Explainability: SHAP values per state for the paper's 6-feature RandomForest.

The RF loses to linear in LOO-CV MAE (0.164 vs 0.143), but it's the right
object for SHAP decomposition — a classic "linear wins at prediction, trees
win at attribution for a modest sample" use case."""
import json
import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

from config import PROCESSED

warnings.filterwarnings("ignore")
TARGET = "conversion_ratio"
# same 6 features as paper_analysis.RF_FEATURES
FEATURES = ["tariff_growth_frac", "annual_ghi", "tariff_2kw_2024",
            "tariff_2kw_2025", "re_2025", "installations"]


def _panel():
    d = pd.read_csv(PROCESSED / "master_state_panel.csv")
    d["tariff_growth_frac"] = (d["tariff_2kw_2025"] - d["tariff_2kw_2024"]) / d["tariff_2kw_2024"]
    return d


def shap_matrix():
    """Returns (states, feature_names, SHAP matrix, base_value, predictions, actuals)."""
    d = _panel().dropna(subset=FEATURES + [TARGET])
    X = d[FEATURES].values
    y = d[TARGET].values
    rf = RandomForestRegressor(n_estimators=400, max_depth=4, random_state=0).fit(X, y)
    expl = shap.TreeExplainer(rf)
    vals = expl.shap_values(X)
    base = float(expl.expected_value)
    preds = rf.predict(X)
    return d["state"].tolist(), FEATURES, vals, base, preds, y


def shap_for_state(state):
    """Per-state SHAP decomposition: feature contributions sorted by |impact|."""
    states, feats, vals, base, preds, y = shap_matrix()
    if state not in states:
        return {"error": f"no SHAP for '{state}' (feature values missing)"}
    i = states.index(state)
    contribs = sorted(
        [{"feature": f, "shap": round(float(vals[i, j]), 4),
          "value": round(float(_panel().set_index("state").loc[state, f]), 3)}
         for j, f in enumerate(feats)],
        key=lambda r: -abs(r["shap"]))
    return {"state": state, "base_value": round(base, 4),
            "prediction": round(float(preds[i]), 4),
            "actual": round(float(y[i]), 4),
            "contributions": contribs}


def dump():
    """Precompute per-state SHAP summary so the dashboard doesn't recompute."""
    states, feats, vals, base, preds, y = shap_matrix()
    out = {"features": feats, "base_value": round(base, 4),
           "by_state": {}}
    pan = _panel().set_index("state")
    for i, st in enumerate(states):
        out["by_state"][st] = {
            "prediction": round(float(preds[i]), 4),
            "actual": round(float(y[i]), 4),
            "shap": {f: round(float(vals[i, j]), 4) for j, f in enumerate(feats)},
            "values": {f: round(float(pan.loc[st, f]), 3) for f in feats},
        }
    # global mean |SHAP| importance
    out["global_importance"] = {
        f: round(float(np.mean(np.abs(vals[:, j]))), 4)
        for j, f in enumerate(feats)
    }
    (PROCESSED / "shap_summary.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    r = dump()
    print("Base conversion (model mean):", r["base_value"])
    print("\nGlobal mean |SHAP|:")
    for f, v in sorted(r["global_importance"].items(), key=lambda x: -x[1]):
        print(f"  {f:25} {v:.4f}")
    print("\nMaharashtra SHAP:")
    print(json.dumps(shap_for_state("Maharashtra"), indent=2))
