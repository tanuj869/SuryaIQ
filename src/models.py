"""Core models the AI tools call:
  1. forecast_re_capacity  — annual RE-capacity series (File 10.2), baselines
  2. conversion model      — tree-ensemble predicting state conversion_ratio
  3. subsidy-paradox test  — subsidy-amount vs implementation-quality features

Everything here is deterministic and runnable without any API key.
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut

from config import PROCESSED, ROOT

warnings.filterwarnings("ignore")
ARTIFACTS = ROOT / "data" / "processed"


# ---------------------------------------------------------------- forecasting
def _panel():
    return pd.read_csv(PROCESSED / "re_capacity_panel.csv")


def forecast_re_capacity(state, horizon=3):
    """Forecast a state's RE-capacity trajectory. Benchmarks naive, linear-trend,
    Holt exponential smoothing, and SARIMAX (per the report's ARIMA/SARIMAX
    requirement); returns the best-available forecast with an interval plus the
    full baseline comparison. Honest about short history (8 annual points)."""
    from statsmodels.tsa.holtwinters import Holt
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    df = _panel()
    s = df[df["state"] == state].sort_values("fy_start")
    if len(s) < 4:
        return {"error": f"insufficient history for {state}"}
    y = s["re_capacity_mw"].to_numpy(dtype=float)
    years = s["fy"].tolist()
    last_fy = int(s["fy_start"].max())

    def wf(fit_fn):  # walk-forward one-step over the last 3 points
        errs = []
        for cut in range(len(y) - 3, len(y)):
            try:
                errs.append(abs(fit_fn(y[:cut]) - y[cut]))
            except Exception:
                return float("nan")
        return float(np.mean(errs))

    def _sarimax_fc(a, h=1):
        m = SARIMAX(a, order=(1, 1, 0), trend="t",
                    enforce_stationarity=False, enforce_invertibility=False)
        return m.fit(disp=False).forecast(h)

    naive = wf(lambda a: a[-1])
    lin = wf(lambda a: np.polyval(np.polyfit(range(len(a)), a, 1), len(a)))
    holt_mae = wf(lambda a: Holt(a, exponential=True).fit().forecast(1)[0])
    sarimax_mae = wf(lambda a: _sarimax_fc(a, 1)[0])

    # final forecast: pick the lowest-backtest-MAE model that fits
    cands = {"naive": naive, "linear": lin, "holt": holt_mae, "sarimax": sarimax_mae}
    best = min((k for k in cands if not np.isnan(cands[k])), key=lambda k: cands[k])
    try:
        if best == "sarimax":
            fc = _sarimax_fc(y, horizon)
        elif best == "holt":
            fc = Holt(y, exponential=True).fit().forecast(horizon)
        elif best == "linear":
            sl, ic = np.polyfit(range(len(y)), y, 1)
            fc = ic + sl * np.arange(len(y), len(y) + horizon)
        else:
            fc = np.repeat(y[-1], horizon)
    except Exception:
        fc = np.repeat(y[-1], horizon)

    resid = float(np.std(np.diff(y)))
    future = [f"{last_fy + i}-{str(last_fy + i + 1)[2:]}" for i in range(1, horizon + 1)]
    return {
        "state": state, "target": "RE installed capacity (MW)",
        "best_model": best,
        "history": {"years": years, "values": [round(v, 1) for v in y]},
        "forecast": {"years": future,
                     "values": [round(float(v), 1) for v in fc],
                     "lower": [round(float(v - 1.96 * resid), 1) for v in fc],
                     "upper": [round(float(v + 1.96 * resid), 1) for v in fc]},
        "backtest_mae": {k: (round(v, 1) if not np.isnan(v) else None)
                         for k, v in cands.items()},
        "note": "RE capacity (MW) incl. large hydro; PMSG rooftop is a subset. "
                "8 annual points only — treat as indicative, not precise.",
    }


def forecast_installations(state, months=12):
    """Project PM Surya Ghar INSTALLATIONS forward (what the report asks for).

    Installations exist only as a single snapshot (no monthly series), so we
    grow the current snapshot using the state's observed RE-capacity CAGR as a
    proxy for momentum. Clearly labelled as an indicative proxy, not a fitted
    monthly time-series forecast."""
    pan = pd.read_csv(PROCESSED / "master_state_panel.csv")
    row = pan[pan["state"] == state]
    if row.empty or pd.isna(row["installations"].iloc[0]):
        return {"error": f"no installation snapshot for {state}"}
    cur = float(row["installations"].iloc[0])
    cagr = row["re_cagr_18_25"].iloc[0]
    cagr = float(cagr) if pd.notna(cagr) else 0.15
    cagr = max(min(cagr, 0.6), 0.0)  # clamp to a sane band
    yrs = months / 12.0
    proj = cur * (1 + cagr) ** yrs
    return {
        "state": state, "target": "PM Surya Ghar installations (cumulative)",
        "current_installations": int(cur), "horizon_months": months,
        "annual_growth_used": round(cagr, 3),
        "projected_installations": int(proj),
        "projected_additional": int(proj - cur),
        "method": "growth proxy (RE-capacity CAGR applied to installation snapshot)",
        "caveat": "Indicative only — installations are a single snapshot, not a "
                  "monthly series; this is a growth projection, not a fitted forecast.",
    }


# --------------------------------------------------------------- conversion ML
SUBSIDY_FEATURES = ["subsidy_per_install_rs", "subsidy_released_cr"]
IMPLQUAL_FEATURES = ["reg_to_app_ratio", "app_per_lakh", "n_districts",
                     "potential_utilization"]
CONTEXT_FEATURES = ["tariff_2kw_2025", "solar_potential_gw", "annual_ghi",
                    "re_share_2025", "re_cagr_18_25"]
ALL_FEATURES = SUBSIDY_FEATURES + IMPLQUAL_FEATURES + CONTEXT_FEATURES
TARGET = "conversion_ratio"


def _xy(cols):
    df = pd.read_csv(PROCESSED / "master_state_panel.csv")
    d = df.dropna(subset=[TARGET]).copy()
    X = d[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    return d, X, d[TARGET].to_numpy()


def _loso(cols, model_fn):
    """Leave-one-state-out CV — honest for N~36."""
    _, X, y = _xy(cols)
    Xv = X.to_numpy()
    preds = np.zeros_like(y)
    for tr, te in LeaveOneOut().split(Xv):
        m = model_fn()
        m.fit(Xv[tr], y[tr])
        preds[te] = m.predict(Xv[te])
    return mean_absolute_error(y, preds), r2_score(y, preds)


def _rf():
    return RandomForestRegressor(n_estimators=400, max_depth=4, random_state=0)


def _gbm():
    # scikit-learn gradient boosting — stands in for XGBoost/LightGBM (the
    # report's named gradient-boosted regressor) with no libomp dependency.
    return HistGradientBoostingRegressor(max_depth=3, max_iter=300,
                                         learning_rate=0.05, random_state=0)


def conversion_model_comparison():
    """Benchmark the report's model set on the conversion target under LOSO CV:
    linear baseline vs RandomForest vs Gradient Boosting (XGBoost stand-in)."""
    out = {}
    for name, fn in [("linear_regression", LinearRegression),
                     ("random_forest", _rf),
                     ("gradient_boosting", _gbm)]:
        mae, r2 = _loso(ALL_FEATURES, fn)
        out[name] = {"loso_mae": round(mae, 4), "loso_r2": round(r2, 3)}
    out["best_model"] = min(out, key=lambda k: out[k]["loso_mae"]
                            if isinstance(out[k], dict) else 1)
    return out


def subsidy_paradox_test():
    """Does implementation-quality beat subsidy-amount at predicting conversion?
    Compares three feature sets under identical LOSO RandomForest CV, and also
    benchmarks the full model set (linear / RF / gradient boosting)."""
    sets = {"subsidy_amount_only": SUBSIDY_FEATURES,
            "implementation_quality_only": IMPLQUAL_FEATURES,
            "subsidy+impl+context (full)": ALL_FEATURES}
    results = {}
    for name, cols in sets.items():
        mae, r2 = _loso(cols, _rf)
        results[name] = {"features": cols, "loso_mae": round(mae, 4),
                         "loso_r2": round(r2, 3)}
    # feature importance on the full model
    d, X, y = _xy(ALL_FEATURES)
    full = _rf().fit(X, y)
    imp = sorted(zip(ALL_FEATURES, full.feature_importances_),
                 key=lambda t: -t[1])
    results["feature_importance_full"] = [{"feature": f, "importance": round(float(i), 3)}
                                          for f, i in imp]
    results["model_comparison_full"] = conversion_model_comparison()
    baseline_mae, baseline_r2 = _loso(ALL_FEATURES, lambda: LinearRegression())
    results["linear_baseline_full"] = {"loso_mae": round(baseline_mae, 4),
                                       "loso_r2": round(baseline_r2, 3)}
    verdict = ("implementation-quality features alone achieve MAE "
               f"{results['implementation_quality_only']['loso_mae']} vs "
               f"{results['subsidy_amount_only']['loso_mae']} for subsidy-amount "
               "alone — " +
               ("supporting" if results['implementation_quality_only']['loso_mae']
                <= results['subsidy_amount_only']['loso_mae'] else "not supporting") +
               " the subsidy-paradox hypothesis at state level (associational).")
    results["verdict"] = verdict
    (ARTIFACTS / "subsidy_paradox_results.json").write_text(json.dumps(results, indent=2))
    return results


def predict_conversion(overrides=None):
    """Train full model on observed data; optionally apply multiplicative
    feature overrides (e.g. {'subsidy_per_install_rs': 0.10}) and re-predict."""
    d, X_obs, y = _xy(ALL_FEATURES)
    m = RandomForestRegressor(n_estimators=400, max_depth=4, random_state=0).fit(X_obs, y)
    X = X_obs.copy()
    if overrides:
        for k, v in overrides.items():
            if k in X.columns:
                X[k] = X[k] * (1 + v)
    return d["state"].tolist(), y, m.predict(X)


if __name__ == "__main__":
    import sys
    print(json.dumps(subsidy_paradox_test(), indent=2))
    print("\nForecast demo (Maharashtra):")
    print(json.dumps(forecast_re_capacity("Maharashtra", 3), indent=2))
