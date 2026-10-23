# Data Dictionary — `master_state_panel`

36 States/UTs x 31 columns. One row per State/UT. Built by `src/build_panel.py` from the 11 raw files.

| Column | Type | Non-null | Description |
|---|---|---|---|
| `state` | object | 36/36 | Canonical State/UT name |
| `region` | object | 35/36 | MNRE region (N/S/E/W/NE) |
| `population_lakh_2011` | float64 | 36/36 | Population in lakh, Census 2011 (for per-lakh rates) |
| `registrations` | int64 | 36/36 | PMSG portal registrations (File 1, Feb 2025) |
| `applications_best` | float64 | 36/36 | Applications; File 4 rollup preferred, else File 1 |
| `installations` | float64 | 36/36 | Installations; File 4 rollup preferred, else File 2 — PRIMARY outcome |
| `households_benefitted` | float64 | 35/36 | Households benefitted (File 1) |
| `households_covered` | float64 | 36/36 | Households covered (File 4) |
| `capacity_mw` | float64 | 36/36 | Installed rooftop capacity, MW (File 4) |
| `subsidy_released_cr` | float64 | 36/36 | CFA subsidy released, Rs crore (File 4, lakh/100) |
| `n_districts` | int64 | 36/36 | Districts reporting in File 4 |
| `conversion_ratio` | float64 | 36/36 | installations / applications — PRIMARY TARGET (subsidy-paradox test) |
| `reg_to_app_ratio` | float64 | 36/36 | applications / registrations (funnel step 1) |
| `app_per_lakh` | float64 | 36/36 | applications / population_lakh |
| `install_per_lakh` | float64 | 36/36 | installations / population_lakh |
| `subsidy_per_install_rs` | float64 | 36/36 | subsidy_released / installations, Rs |
| `tariff_2kw_2024` | float64 | 35/36 | Avg residential 2kW tariff 2024, Rs/kWh (File 5.1) |
| `tariff_2kw_2025` | float64 | 35/36 | Avg residential 2kW tariff 2025, Rs/kWh (File 5.2) |
| `tariff_change_pct` | float64 | 35/36 | % change tariff 2024->2025 |
| `solar_potential_gw` | float64 | 35/36 | State solar potential @6.69% wasteland, GW (File 6) |
| `potential_utilization` | float64 | 35/36 | capacity_mw / (potential_gw*1000) |
| `annual_ghi` | float64 | 28/36 | Annual avg GHI kWh/m2/day (File 9, NASA POWER) |
| `re_2025` | float64 | 35/36 | RE installed capacity 31-Mar-2025, MW (File 10.1) |
| `re_share_2025` | float64 | 35/36 | RE / (RE+NonRE) 2025 (maturity proxy) |
| `re_cagr_18_25` | float64 | 34/36 | RE capacity CAGR 2018->2025 |
| `payback_attractiveness` | float64 | 36/36 | Engineered proxy: tariff x GHI x subsidy/install |
| `expected_install_per_lakh` | float64 | 36/36 | OLS-predicted install_per_lakh from potential+tariff+GHI |
| `adoption_gap` | float64 | 36/36 | install_per_lakh - expected (negative = underperforming) |
| `suitability_index` | float64 | 36/36 | 0-100 composite: potential, GHI, tariff, RE share |
| `installations_f4` | float64 | 36/36 | Raw File 4 installations (reconciliation) |
| `installations_file2` | float64 | 35/36 | Raw File 2 installations (reconciliation) |