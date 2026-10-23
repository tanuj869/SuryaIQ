"""SuryaIQ configuration: paths, canonical state names, reference tables.

All reference numbers that are NOT in the 11 provided files (population) are
sourced explicitly below so the dataset register can cite them. Where a value
cannot be sourced it is left out rather than invented.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
for d in (INTERIM, PROCESSED):
    d.mkdir(parents=True, exist_ok=True)

# Raw file names (exact, as shipped in the capstone zip)
FILES = {
    "reg_app_hh": "1. Registrations_Applications_Households_2025-02-06.xlsx",
    "installs": "2. Installations_27-11-2024.xlsx",
    "mh_districts": "3. District-wise beneficiaries (Maharashtra)_11-12-2024.xlsx",
    "district_discom": "4. District_Discom_Application_Installation_Ratio_PMSG_23-08-2026.xlsx.xlsx",
    "tariff_2024": "5.1 PMSG Tariff - 2024.xlsx",
    "tariff_2025": "5.2 PMSG Tariff - 2025.xlsx",
    "solar_potential": "6. Solar potential.xlsx",
    "irradiance": "9. Solar irradiance.xlsx",
    "re_two_points": "10.1 state-wise RE and non-RE installed capacity at two points.xlsx",
    "re_panel": "10.2 MNRE_RE_Capacity_AnnualPanel_2017-18_to_2024-25.xlsx",
}

# --- Canonical state-name normalization -------------------------------------
# Maps every spelling/casing variant seen across the 11 files to one canonical
# name. Keys are lowercase, stripped, punctuation-light.
_CANON = {
    "andaman & nicobar island": "Andaman & Nicobar",
    "andaman & nicobar islands": "Andaman & Nicobar",
    "andaman and nicobar": "Andaman & Nicobar",
    "andaman and nicobar islands": "Andaman & Nicobar",
    "andman and nicobar islands": "Andaman & Nicobar",
    "andhra pradesh": "Andhra Pradesh",
    "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chandigarh": "Chandigarh",
    "chhattisgarh": "Chhattisgarh",
    "chattisgarh": "Chhattisgarh",
    "dadra & nagar haveli and daman & diu": "Dadra & Nagar Haveli and Daman & Diu",
    "dadra & nagar haveli & daman &diu": "Dadra & Nagar Haveli and Daman & Diu",
    "dadra and nagar haveli and daman and diu": "Dadra & Nagar Haveli and Daman & Diu",
    "dadra and nagar haveli": "Dadra & Nagar Haveli and Daman & Diu",
    "daman & diu": "Dadra & Nagar Haveli and Daman & Diu",
    "delhi": "Delhi",
    "nct of delhi": "Delhi",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh",
    "jammu & kashmir": "Jammu & Kashmir",
    "jammu and kashmir": "Jammu & Kashmir",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "ladakh": "Ladakh",
    "lakshadweep": "Lakshadweep",
    "madhya pradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Odisha",
    "orissa": "Odisha",
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tamil nadu": "Tamil Nadu",
    "telangana": "Telangana",
    "telengana": "Telangana",
    "tripura": "Tripura",
    "uttar pradesh": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand",
    "uttaranchal": "Uttarakhand",
    "west bengal": "West Bengal",
}


def canon_state(name):
    """Return the canonical state/UT name, or None for blanks/non-states.

    Strips qualifier suffixes used in the tariff files, e.g.
    'Bihar (Urban Areas)', 'Uttar Pradesh (URBAN)', 'Maharashtra - Mumbai-(TATA's)',
    'Delhi-(NDMC)' -> the base state name.
    """
    if name is None:
        return None
    key = " ".join(str(name).strip().lower().split())
    # drop parenthetical and dash qualifiers before matching
    for sep in (" (", "(", " - ", "-("):
        if sep in key:
            key = key.split(sep)[0].strip()
    if key in ("", "total", "grand total", "others", "north", "south", "east",
               "west", "north east", "nan", "state", "state/ut", "d.v.c.", "dvc"):
        return None
    return _CANON.get(key, str(name).strip())


# --- State population (for per-lakh normalization) ---------------------------
# Source: Census of India 2011, state/UT totals (persons), in LAKH (1 lakh =
# 100,000). Used only to derive per-lakh adoption rates. Dated 2011 — flagged in
# the data dictionary as a known limitation (no newer official state census).
POP_LAKH_2011 = {
    "Andhra Pradesh": 843.0, "Arunachal Pradesh": 13.8, "Assam": 312.1,
    "Bihar": 1040.9, "Chhattisgarh": 255.4, "Goa": 14.6, "Gujarat": 604.4,
    "Haryana": 253.5, "Himachal Pradesh": 68.6, "Jammu & Kashmir": 125.4,
    "Jharkhand": 330.0, "Karnataka": 610.9, "Kerala": 334.1,
    "Madhya Pradesh": 726.3, "Maharashtra": 1123.7, "Manipur": 28.6,
    "Meghalaya": 29.7, "Mizoram": 10.9, "Nagaland": 19.8, "Odisha": 419.7,
    "Punjab": 277.4, "Rajasthan": 685.5, "Sikkim": 6.1, "Tamil Nadu": 721.5,
    "Telangana": 350.0, "Tripura": 36.7, "Uttar Pradesh": 1998.1,
    "Uttarakhand": 100.9, "West Bengal": 913.5, "Delhi": 167.9,
    "Puducherry": 12.5, "Chandigarh": 10.5, "Andaman & Nicobar": 3.8,
    "Dadra & Nagar Haveli and Daman & Diu": 5.9, "Ladakh": 2.7,
    "Lakshadweep": 0.6,
}
