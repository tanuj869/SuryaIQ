"""Merge all interim tables into one canonical state-level analytical table,
engineer features, and write processed outputs + a reconciliation report."""
import numpy as np
import pandas as pd

import ingest
from config import PROCESSED, ROOT, POP_LAKH_2011


def build():
    reg = ingest.reg_app_hh()
    f2 = ingest.installs()
    f4 = ingest.district_discom_rollup()
    t24 = ingest.tariff(2024)
    t25 = ingest.tariff(2025)
    pot = ingest.solar_potential()
    irr = ingest.irradiance()
    rec = ingest.re_capacity()

    df = reg.merge(f4, on="state", how="outer") \
            .merge(f2, on="state", how="outer") \
            .merge(t24, on="state", how="left") \
            .merge(t25, on="state", how="left") \
            .merge(pot, on="state", how="left") \
            .merge(irr, on="state", how="left") \
            .merge(rec, on="state", how="left")

    df["population_lakh_2011"] = df["state"].map(POP_LAKH_2011)

    # --- primary target & funnel ratios (file 4 is the preferred source) -----
    df["installations"] = df["installations_f4"].fillna(df["installations_file2"])
    df["applications_best"] = df["applications_f4"].fillna(df["applications"])
    df["conversion_ratio"] = df["installations"] / df["applications_best"]
    df["reg_to_app_ratio"] = df["applications"] / df["registrations"]
    df["app_to_install_ratio"] = df["installations"] / df["applications_best"]

    # --- normalized adoption -------------------------------------------------
    df["app_per_lakh"] = df["applications_best"] / df["population_lakh_2011"]
    df["install_per_lakh"] = df["installations"] / df["population_lakh_2011"]

    # --- economics / resource ------------------------------------------------
    df["subsidy_per_install_rs"] = (df["subsidy_released_cr"] * 1e7) / df["installations"]
    df["tariff_change_pct"] = (df["tariff_2kw_2025"] - df["tariff_2kw_2024"]) / df["tariff_2kw_2024"] * 100
    df["potential_utilization"] = df["capacity_mw"] / (df["solar_potential_gw"] * 1000)
    # payback proxy: higher tariff + higher subsidy/install + more sun => faster payback (higher=better)
    df["payback_attractiveness"] = (
        df["tariff_2kw_2025"].fillna(df["tariff_2kw_2025"].median())
        * df["annual_ghi"].fillna(df["annual_ghi"].median())
        * (df["subsidy_per_install_rs"].fillna(df["subsidy_per_install_rs"].median()) / 1e5)
    )

    # --- adoption gap (actual vs expected given potential, tariff, sun) -------
    base = df.dropna(subset=["install_per_lakh", "solar_potential_gw",
                             "tariff_2kw_2025", "annual_ghi"]).copy()
    if len(base) > 8:
        from numpy.polynomial import polynomial as P  # noqa
        X = base[["solar_potential_gw", "tariff_2kw_2025", "annual_ghi"]].values
        y = base["install_per_lakh"].values
        X1 = np.column_stack([np.ones(len(X)), X])
        coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
        allX = df[["solar_potential_gw", "tariff_2kw_2025", "annual_ghi"]]
        pred = coef[0] + allX.fillna(allX.median()).values @ coef[1:]
        df["expected_install_per_lakh"] = pred
        df["adoption_gap"] = df["install_per_lakh"] - df["expected_install_per_lakh"]

    # --- rooftop suitability index (0-100 composite) -------------------------
    # Weighted min-max of four drivers; robust to missing components by
    # renormalizing weights over whatever is available per state.
    def mm(col):
        s = pd.to_numeric(df[col], errors="coerce")
        return (s - s.min()) / (s.max() - s.min())
    comp = {"solar_potential_gw": 0.35, "annual_ghi": 0.25,
            "tariff_2kw_2025": 0.20, "re_share_2025": 0.20}
    norm = pd.DataFrame({c: mm(c) for c in comp})
    w = pd.Series(comp)
    wsum = norm.notna().mul(w, axis=1).sum(axis=1)
    df["suitability_index"] = (norm.mul(w, axis=1).sum(axis=1) / wsum.replace(0, np.nan)) * 100

    # order columns
    cols = ["state", "region", "population_lakh_2011", "registrations",
            "applications_best", "installations", "households_benefitted",
            "households_covered", "capacity_mw", "subsidy_released_cr",
            "n_districts", "conversion_ratio", "reg_to_app_ratio",
            "app_per_lakh", "install_per_lakh", "subsidy_per_install_rs",
            "tariff_2kw_2024", "tariff_2kw_2025", "tariff_change_pct",
            "solar_potential_gw", "potential_utilization", "annual_ghi",
            "re_2025", "re_share_2025", "re_cagr_18_25",
            "payback_attractiveness", "expected_install_per_lakh",
            "adoption_gap", "suitability_index",
            "installations_f4", "installations_file2"]
    df = df[[c for c in cols if c in df.columns]].sort_values("state").reset_index(drop=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "master_state_panel.csv", index=False)
    try:
        df.to_parquet(PROCESSED / "master_state_panel.parquet", index=False)
    except Exception:
        pass
    ingest.re_panel().to_csv(PROCESSED / "re_capacity_panel.csv", index=False)

    _reconciliation(df)
    return df


def _reconciliation(df):
    """File 2 vs File 4 installation discrepancy report."""
    r = df.dropna(subset=["installations_f4", "installations_file2"]).copy()
    r["abs_diff"] = (r["installations_f4"] - r["installations_file2"]).abs()
    r["pct_diff"] = r["abs_diff"] / r["installations_file2"].replace(0, np.nan) * 100
    r = r[["state", "installations_file2", "installations_f4", "abs_diff", "pct_diff"]]
    r = r.sort_values("pct_diff", ascending=False)
    lines = [
        "# Reconciliation Report — Installation counts across sources", "",
        "File 2 (`Installations_27-11-2024`, snapshot Nov 2024) vs File 4 "
        "(`District_Discom...`, district rollup, later snapshot). These are "
        "different snapshot dates, so File 4 ≥ File 2 is expected as the scheme "
        "progressed. File 4 is used as the canonical source; File 2 is kept for "
        "cross-check only.", "",
        r.round(2).to_markdown(index=False),
    ]
    (ROOT / "docs" / "reconciliation_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    d = build()
    print("master_state_panel:", d.shape)
    print("states:", d["state"].nunique())
    print(d[["state", "applications_best", "installations", "conversion_ratio",
             "suitability_index"]].round(3).to_string(index=False))
