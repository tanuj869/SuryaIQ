# Dataset Register — SuryaIQ

Inventory of every raw source. **Grain and snapshot date vary** — PMSG data is
published in batches, not via a real-time API. File 4 is the richest and most
recent PMSG source and is treated as canonical for applications/installations/
subsidy/capacity.

| # | File | Source / portal | Snapshot | Grain | Key variables | Notes |
|---|------|-----------------|----------|-------|---------------|-------|
| 1 | Registrations_Applications_Households | data.gov.in (RS Session 267) | 2025-02-06 | State/UT | registrations, applications, households benefitted | Funnel top. `NA` for small UTs. |
| 2 | Installations | data.gov.in (RS Session 266) | 2024-11-27 | State/UT | installations | Early snapshot; cross-check only. |
| 3 | District-wise beneficiaries (Maharashtra) | data.gov.in (RS Session 266) | 2024-12-11 | District (MH) | beneficiaries w/ CFA | Geo case study. |
| 4 | District_Discom_Application_Installation_Ratio | PMSG portal (District + DISCOM) | 2026-08-23 (filename) | District & DISCOM | applications, installations, households, capacity MW, **subsidy released (lakh)** | **Canonical.** Rolled up to state. 803 district rows, 36 states. |
| 5.1 | PMSG Tariff 2024 | SERC tariff orders | FY2023-24 eff. | Utility × load slab | Av. rate, duty, total (Paise/kWh) | Used the **Domestic 2 kW** slab → Rs/kWh. |
| 5.2 | PMSG Tariff 2025 | SERC tariff orders | FY2024-25 eff. | Utility × load slab | same | Urban/Rural/city variants averaged to state. |
| 6 | Solar potential | NITI ICeD / MNRE | static | State | potential @6.69% & @3% wasteland (GW) | Used 6.69% figure. |
| 7 | ToD regime type (PDF) | MNRE / SERC | 2025 | State (doc) | net-metering / net-billing / ToD regime | Forward-looking feature; extract via RAG. |
| 9 | Solar irradiance | NASA POWER climatology | climatology | State (1 city) × month | GHI (ALLSKY_SFC_SW_DWN) monthly + annual | Header at row 12. 28 states covered; 8 small UTs absent. |
| 10.1 | RE / Non-RE capacity at two points | MNRE physical progress | 2018 & 2025 | State × 2 points | Non-RE/RE MW | → RE share + CAGR. |
| 10.2 | MNRE RE Capacity Annual Panel | MNRE physical progress | 2017-18 → 2024-25 | **State × year** | annual RE capacity MW | **Only true time series** → `re_capacity_panel.csv` for forecasting baselines. |

## External reference (not in the zip)
- **State population** — Census of India 2011 state/UT totals, embedded in `src/config.py` (`POP_LAKH_2011`), used only for per-lakh normalization. **Limitation:** dated 2011; no newer official state census.

## Known data-quality issues (carried into `docs/reconciliation_report.md`)
- Files 1, 2, 4 report different installation counts for the same state — different snapshot dates. File 4 is canonical; File 2 kept for cross-check.
- Irradiance covers 28 states (one representative city each); 8 small UTs are NaN.
- PMSG launched Feb 2024 → short history; design favors **state cross-section + short panel**, not long univariate forecasting (reserved for File 10.2).
