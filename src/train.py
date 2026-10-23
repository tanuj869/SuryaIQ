"""Train the full SuryaIQ model set on ALL datasets and write a training
report (per the AI Summary Report: ARIMA/SARIMAX + gradient-boosted models,
trained across the merged multi-source dataset).

Datasets exercised (all 11 raw files flow in via build_panel):
  - Conversion model (classification/regression target): files 1,2,4 (funnel),
    5.1/5.2 (tariff), 6 (potential), 9 (irradiance), 10.1 (RE maturity)
  - Time-series forecast: file 10.2 (RE-capacity annual panel)
  - Installation projection: files 2/4 snapshot + 10.1 growth
Outputs: data/processed/training_report.json
"""
import json

import numpy as np
import pandas as pd

import models
from config import PROCESSED


DATASET_USAGE = {
    "1 Registrations/Applications/Households": "funnel + reg_to_app ratio (features)",
    "2 Installations": "installation target (reconciled with file 4)",
    "4 District/DISCOM Application-Installation": "canonical applications/installations/subsidy/capacity target + features",
    "5.1/5.2 Tariff 2024/2025": "tariff feature + tariff-change + payback proxy",
    "6 Solar potential": "potential-utilization + suitability (feature)",
    "9 Solar irradiance": "annual GHI + seasonality (feature)",
    "10.1 RE/Non-RE two points": "RE share + CAGR (features)",
    "10.2 RE-capacity annual panel": "SARIMAX/Holt time-series forecast target",
    "7 ToD regime (PDF)": "RAG corpus (image-based; OCR pending)",
}


def main():
    panel = pd.read_csv(PROCESSED / "master_state_panel.csv")
    report = {"n_states": int(panel["state"].nunique()),
              "dataset_usage": DATASET_USAGE, "models": {}}

    # 1) Conversion models (report's full model set under LOSO CV)
    paradox = models.subsidy_paradox_test()
    report["models"]["conversion_regression"] = {
        "target": "conversion_ratio (installations/applications)",
        "features_used": models.ALL_FEATURES,
        "model_comparison_LOSO": paradox["model_comparison_full"],
        "feature_importance": paradox["feature_importance_full"][:6],
        "subsidy_paradox_verdict": paradox["verdict"],
    }

    # 2) Classification: top-quartile adopter (report mentions classification)
    report["models"]["topquartile_classification"] = _classify_topquartile(panel)

    # 3) Time-series forecast benchmarks on the RE-capacity panel (5 states + note)
    demo = {}
    for st in ["Maharashtra", "Gujarat", "Rajasthan", "Kerala", "Uttar Pradesh"]:
        fc = models.forecast_re_capacity(st, 3)
        if "backtest_mae" in fc:
            demo[st] = {"best_model": fc["best_model"], "backtest_mae": fc["backtest_mae"]}
    report["models"]["re_capacity_forecast"] = {
        "target": "RE installed capacity (MW), annual",
        "benchmarks": ["naive", "linear", "holt", "sarimax"],
        "per_state": demo,
    }

    # 4) Installation projection demo
    report["models"]["installation_projection"] = {
        st: models.forecast_installations(st, 12)
        for st in ["Maharashtra", "Bihar", "Rajasthan"]}

    (PROCESSED / "training_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _classify_topquartile(panel):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import LeaveOneOut
    d = panel.dropna(subset=["conversion_ratio"]).copy()
    thr = d["conversion_ratio"].quantile(0.75)
    y = (d["conversion_ratio"] >= thr).astype(int).to_numpy()
    X = d[models.ALL_FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).to_numpy()
    preds = np.zeros_like(y)
    for tr, te in LeaveOneOut().split(X):
        m = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=0)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return {"target": "top-quartile conversion state (>= p75)",
            "loso_accuracy": round(float(accuracy_score(y, preds)), 3),
            "loso_f1": round(float(f1_score(y, preds, zero_division=0)), 3),
            "positive_rate": round(float(y.mean()), 3)}


if __name__ == "__main__":
    r = main()
    print("=== SuryaIQ training report ===")
    print("States:", r["n_states"])
    cm = r["models"]["conversion_regression"]["model_comparison_LOSO"]
    print("\nConversion model comparison (LOSO MAE):")
    for k, v in cm.items():
        if isinstance(v, dict):
            print(f"  {k:20} MAE={v['loso_mae']}  R2={v['loso_r2']}")
    print("  best:", cm["best_model"])
    print("\nTop-quartile classifier:", r["models"]["topquartile_classification"])
    print("\nForecast best-model per state:")
    for st, info in r["models"]["re_capacity_forecast"]["per_state"].items():
        print(f"  {st:16} {info['best_model']}  {info['backtest_mae']}")
    print("\nInstallation projection (Maharashtra, 12mo):",
          r["models"]["installation_projection"]["Maharashtra"]["projected_installations"])
    print("\nWrote data/processed/training_report.json")
