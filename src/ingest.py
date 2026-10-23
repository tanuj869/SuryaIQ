"""Per-file readers/cleaners. Each returns a tidy, state-keyed DataFrame and
writes an interim CSV. Robust to the quirks found in the raw files (blank
header rows, merged state cells, casing, 'NA' strings)."""
import numpy as np
import pandas as pd

from config import FILES, RAW, INTERIM, canon_state


def _na(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    return np.nan if s.upper() in ("NA", "N/A", "", "-") else v


def reg_app_hh():
    df = pd.read_excel(RAW / FILES["reg_app_hh"])
    df.columns = ["sl", "state", "registrations", "applications", "households_benefitted"]
    df["state"] = df["state"].map(canon_state)
    df = df.dropna(subset=["state"]).copy()
    for c in ["registrations", "applications", "households_benefitted"]:
        df[c] = pd.to_numeric(df[c].map(_na), errors="coerce")
    out = df[["state", "registrations", "applications", "households_benefitted"]]
    out.to_csv(INTERIM / "reg_app_hh.csv", index=False)
    return out


def installs():
    df = pd.read_excel(RAW / FILES["installs"])
    df.columns = ["sl", "state", "installations_file2"]
    df["state"] = df["state"].map(canon_state)
    df = df.dropna(subset=["state"]).copy()
    df["installations_file2"] = pd.to_numeric(df["installations_file2"].map(_na), errors="coerce")
    out = df[["state", "installations_file2"]]
    out.to_csv(INTERIM / "installs_file2.csv", index=False)
    return out


def district_discom_rollup():
    """File 4: roll district rows up to state. ffill merged State cells."""
    df = pd.read_excel(RAW / FILES["district_discom"], sheet_name="District info")
    df.columns = ["state", "district", "applications", "installations",
                  "households_covered", "capacity_mw", "subsidy_lakh"]
    df["state"] = df["state"].ffill().map(canon_state)
    df = df.dropna(subset=["state", "district"]).copy()
    for c in ["applications", "installations", "households_covered", "capacity_mw", "subsidy_lakh"]:
        df[c] = pd.to_numeric(df[c].map(_na), errors="coerce")
    # keep district-level table too (for Maharashtra / geo drilldowns)
    df.to_csv(INTERIM / "file4_districts.csv", index=False)
    agg = (df.groupby("state", as_index=False)
             .agg(applications_f4=("applications", "sum"),
                  installations_f4=("installations", "sum"),
                  households_covered=("households_covered", "sum"),
                  capacity_mw=("capacity_mw", "sum"),
                  subsidy_lakh=("subsidy_lakh", "sum"),
                  n_districts=("district", "nunique")))
    agg["subsidy_released_cr"] = agg["subsidy_lakh"] / 100.0  # lakh -> crore
    agg.to_csv(INTERIM / "file4_state_rollup.csv", index=False)
    return agg


# DISCOM-specific utility rows excluded from state-level tariff aggregation
# (per the paper: avoids double-counting households already in the state figure).
_DISCOM_MARKERS = r"Torrent|B\.E\.S\.T|BEST|Adani|TATA|D\.V\.C|DVC"


def tariff(year):
    """Representative residential tariff = Domestic 2 kW slab, Total P/kWh -> Rs/kWh.
    Averages the plain state + Urban/Rural rows; excludes DISCOM-specific circles
    (Torrent, Mumbai BEST/Adani/TATA, DVC) to match the paper's panel build."""
    f = FILES[f"tariff_{year}"]
    sheet = "Domestic - 2KW" if year == 2024 else "Domestic - 2 KW"
    df = pd.read_excel(RAW / f, sheet_name=sheet)
    df = df.iloc[:, :6]
    df.columns = ["sr", "utility", "eff_from", "avg_rate", "duty_tax", "total"]
    # drop DISCOM-specific rows before mapping to state
    mask = df["utility"].astype(str).str.contains(_DISCOM_MARKERS, case=False,
                                                  regex=True, na=False)
    df = df[~mask].copy()
    df["state"] = df["utility"].map(canon_state)
    df = df.dropna(subset=["state"]).copy()
    df["total"] = pd.to_numeric(df["total"].map(_na), errors="coerce")
    df[f"tariff_2kw_{year}"] = df["total"] / 100.0  # paise -> Rs/kWh
    out = df.groupby("state", as_index=False)[f"tariff_2kw_{year}"].mean()
    out.to_csv(INTERIM / f"tariff_{year}.csv", index=False)
    return out


def solar_potential():
    df = pd.read_excel(RAW / FILES["solar_potential"])
    df.columns = ["state", "region", "potential_gw_669", "potential_gw_3"]
    df["state"] = df["state"].map(canon_state)
    df = df.dropna(subset=["state"]).copy()
    df["solar_potential_gw"] = pd.to_numeric(df["potential_gw_669"].map(_na), errors="coerce")
    out = df[["state", "region", "solar_potential_gw"]]
    out.to_csv(INTERIM / "solar_potential.csv", index=False)
    return out


def irradiance():
    df = pd.read_excel(RAW / FILES["irradiance"], header=11)
    df = df.rename(columns={df.columns[1]: "state", "ANN": "annual_ghi"})
    df["state"] = df["state"].map(canon_state)
    df = df.dropna(subset=["state"]).copy()
    df["annual_ghi"] = pd.to_numeric(df["annual_ghi"], errors="coerce")
    months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    monthly = df[["state"] + [m for m in months if m in df.columns]].copy()
    monthly.to_csv(INTERIM / "irradiance_monthly.csv", index=False)
    out = df[["state", "annual_ghi"]]
    out.to_csv(INTERIM / "irradiance.csv", index=False)
    return out


def re_capacity():
    """File 10.1: RE / Non-RE at 2018 & 2025 -> maturity + CAGR."""
    df = pd.read_excel(RAW / FILES["re_two_points"])
    df.columns = ["sr", "state", "nonre_2018", "re_2018", "nonre_2025", "re_2025"]
    df["state"] = df["state"].map(canon_state)
    df = df.dropna(subset=["state"]).copy()
    for c in ["nonre_2018", "re_2018", "nonre_2025", "re_2025"]:
        df[c] = pd.to_numeric(df[c].map(_na), errors="coerce")
    df["re_share_2025"] = df["re_2025"] / (df["re_2025"] + df["nonre_2025"])
    df["re_cagr_18_25"] = (df["re_2025"] / df["re_2018"].replace(0, np.nan)) ** (1/7) - 1
    out = df[["state", "re_2018", "re_2025", "re_share_2025", "re_cagr_18_25"]]
    out.to_csv(INTERIM / "re_capacity.csv", index=False)
    return out


def re_panel():
    """File 10.2 Sheet2: tidy State x Year RE-capacity annual panel (MW)."""
    df = pd.read_excel(RAW / FILES["re_panel"], sheet_name="Sheet2")
    year_cols = [c for c in df.columns if isinstance(c, str) and "-" in c]
    df = df.rename(columns={df.columns[0]: "state"})
    df["state"] = df["state"].map(canon_state)
    df = df.dropna(subset=["state"]).copy()
    long = df.melt(id_vars="state", value_vars=year_cols,
                   var_name="fy", value_name="re_capacity_mw")
    long["re_capacity_mw"] = pd.to_numeric(long["re_capacity_mw"], errors="coerce")
    long = long.dropna(subset=["re_capacity_mw"])
    long["fy_start"] = long["fy"].str[:4].astype(int)
    long = long.sort_values(["state", "fy_start"])
    long.to_csv(INTERIM / "re_capacity_panel.csv", index=False)
    return long


if __name__ == "__main__":
    for fn in (reg_app_hh, installs, district_discom_rollup, solar_potential,
               irradiance, re_capacity, re_panel):
        fn()
    tariff(2024); tariff(2025)
    print("ingest: interim CSVs written to", INTERIM)
