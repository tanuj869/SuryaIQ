"""Policy scenario simulations (ASSOCIATIONAL, not causal).

Every result is "predicted under model assumptions". Elasticities come from
cross-state variation, which does not identify a causal effect.
"""
import numpy as np
import pandas as pd

from config import PROCESSED
import models


def _panel():
    return pd.read_csv(PROCESSED / "master_state_panel.csv")


def _cross_elasticity(x_col, y_col="conversion_ratio"):
    """log-log OLS slope across states as a crude elasticity."""
    d = _panel().dropna(subset=[x_col, y_col])
    d = d[(d[x_col] > 0) & (d[y_col] > 0)]
    if len(d) < 8:
        return 0.0
    b = np.polyfit(np.log(d[x_col]), np.log(d[y_col]), 1)[0]
    return float(b)


def scenario(name="base", subsidy_pct=0.0, tariff_pct=0.0, target_states=None):
    """Simulate a policy scenario and return state-level predicted conversion
    plus the implied change in installations, holding applications fixed.

    Tariff scenarios are routed through the paper's 2-feature LINEAR model on
    TARIFF GROWTH (coefficient ~1.24) — the paper's documented mechanism. A
    tariff_pct of 0.10 is interpreted as a +10 percentage-point tariff-growth
    shock, consistent with the paper's Scenario D framing."""
    d = _panel().dropna(subset=["conversion_ratio", "applications_best"]).copy()
    states, actual, base_pred = models.predict_conversion()
    base = pd.DataFrame({"state": states, "conv_pred_base": base_pred})

    # Tariff scenario via the paper's linear coefficient (uniform additive lift)
    tariff_lift = 0.0
    if tariff_pct:
        try:
            import json
            from config import PROCESSED as _PROC
            coef = json.loads((_PROC / "paper_results.json").read_text()) \
                ["table2_model_comparison"]["equation"]["coef_tariff_growth"]
            # tariff_pct expressed as a fraction (0.10 == +10 pp tariff-growth shock)
            tariff_lift = float(coef) * tariff_pct
        except Exception:
            tariff_lift = 0.12 * tariff_pct * 10  # safe fallback (~0.12 per pp)
    overrides = {}
    if subsidy_pct:
        overrides["subsidy_per_install_rs"] = subsidy_pct
    _, _, scen_pred = models.predict_conversion(overrides or None)
    base["conv_pred_scenario"] = np.clip(scen_pred + tariff_lift, 0.0, 1.0)

    out = d.merge(base, on="state", how="inner")
    if target_states:  # apply uplift only to targeted states (others unchanged)
        mask = ~out["state"].isin(target_states)
        out.loc[mask, "conv_pred_scenario"] = out.loc[mask, "conv_pred_base"]

    out["installs_base"] = out["conv_pred_base"] * out["applications_best"]
    out["installs_scenario"] = out["conv_pred_scenario"] * out["applications_best"]
    out["delta_installs"] = out["installs_scenario"] - out["installs_base"]
    out["subsidy_burden_cr"] = out["subsidy_released_cr"].fillna(0) * (1 + subsidy_pct)

    summary = {
        "scenario": name,
        "params": {"subsidy_pct": subsidy_pct, "tariff_pct": tariff_pct,
                   "target_states": target_states},
        "national_installs_base": int(out["installs_base"].sum()),
        "national_installs_scenario": int(out["installs_scenario"].sum()),
        "national_delta_installs": int(out["delta_installs"].sum()),
        "pct_change": round(out["delta_installs"].sum() /
                            max(out["installs_base"].sum(), 1) * 100, 2),
        "total_subsidy_burden_cr": round(out["subsidy_burden_cr"].sum(), 1),
        "elasticity_conv_wrt_subsidy_per_install": round(
            _cross_elasticity("subsidy_per_install_rs"), 3),
        "elasticity_conv_wrt_tariff": round(_cross_elasticity("tariff_2kw_2025"), 3),
        "disclaimer": "Predicted under model assumptions. Associational, NOT causal.",
    }
    detail = out[["state", "applications_best", "conv_pred_base",
                  "conv_pred_scenario", "delta_installs"]].round(3)
    return summary, detail


# Canonical scenario set (build-prompt §4 Step 7)
def run_all():
    pan = _panel()
    median_pot = pan["solar_potential_gw"].median()
    median_conv = pan["conversion_ratio"].median()
    high_pot_low_adopt = pan[(pan["solar_potential_gw"] > median_pot) &
                             (pan["conversion_ratio"] < median_conv)]["state"].tolist()
    specs = {
        "A_base": dict(),
        "B_subsidy_+10%": dict(subsidy_pct=0.10),
        "C_tariff_+10%": dict(tariff_pct=0.10),
        "D_target_high_potential_low_adoption": dict(subsidy_pct=0.10,
                                                     target_states=high_pot_low_adopt),
        "E_combined": dict(subsidy_pct=0.10, tariff_pct=0.10,
                           target_states=high_pot_low_adopt),
    }
    rows = []
    for name, kw in specs.items():
        s, _ = scenario(name=name, **kw)
        rows.append(s)
    return rows, high_pot_low_adopt


if __name__ == "__main__":
    import json
    rows, targets = run_all()
    print("High-potential low-adoption targets:", targets)
    for r in rows:
        print(f"\n{r['scenario']}: installs {r['national_installs_base']:,} -> "
              f"{r['national_installs_scenario']:,} ({r['pct_change']:+.1f}%), "
              f"subsidy burden Rs{r['total_subsidy_burden_cr']:,.0f} cr")
