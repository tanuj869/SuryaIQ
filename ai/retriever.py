"""Vectorless RAG — BM25 + keyword + metadata over a lightweight corpus.

Corpus (no vector DB, no embeddings, no hosting cost):
  - project markdown docs (register, dictionary, reconciliation, README)
  - one auto-generated fact sheet per State/UT (from the master panel)
  - the literature-synthesis workbook sheets (text)

Rationale (per the AI Summary Report): for ~dozens of docs this is easier,
free, faster, and lower-dependency than a vector database.
"""
import re
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def _tok(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _state_factsheets():
    df = pd.read_csv(PROCESSED / "master_state_panel.csv")
    docs = []
    for _, r in df.iterrows():
        s = r["state"]
        parts = [f"State: {s}."]
        fmt = {
            "applications_best": "applications {:.0f}",
            "installations": "installations {:.0f}",
            "conversion_ratio": "application-to-installation conversion ratio {:.3f}",
            "subsidy_released_cr": "subsidy released Rs {:.0f} crore",
            "capacity_mw": "installed capacity {:.1f} MW",
            "tariff_2kw_2025": "residential tariff {:.2f} Rs/kWh (2025)",
            "solar_potential_gw": "solar potential {:.1f} GW",
            "annual_ghi": "annual irradiance {:.2f} kWh/m2/day",
            "re_share_2025": "renewable share {:.2f}",
            "suitability_index": "rooftop suitability index {:.1f}/100",
            "adoption_gap": "adoption gap {:.2f} (negative = underperforming)",
        }
        for col, tmpl in fmt.items():
            v = r.get(col)
            if pd.notna(v):
                parts.append(tmpl.format(v))
        docs.append({"id": f"state::{s}", "source": f"factsheet:{s}",
                     "type": "state_factsheet", "text": " ".join(parts)})
    return docs


# meta/spec docs that are retrieval noise for substantive questions
_SKIP_DOCS = {"SuryaIQ_Master_Build_Prompt.md", "suryaiq_vs_chatgpt.md"}


def _markdown_docs():
    docs = []
    for p in (ROOT / "docs").glob("*.md"):
        if p.name in _SKIP_DOCS:
            continue
        docs.append({"id": f"doc::{p.name}", "source": p.name, "type": "doc",
                     "text": p.read_text(errors="ignore")})
    return docs


def _briefing_docs():
    """Synthesized, substantive briefing notes built from the data so the RAG
    has real PM Surya Ghar content to retrieve and cite (not just metadata)."""
    df = pd.read_csv(PROCESSED / "master_state_panel.csv")
    docs = []
    apps, inst = df["applications_best"].sum(), df["installations"].sum()
    top_inst = df.nlargest(5, "installations")["state"].tolist()
    top_conv = df.nlargest(5, "conversion_ratio")["state"].tolist()
    low_conv = df.nsmallest(5, "conversion_ratio")["state"].tolist()
    docs.append({"id": "brief::national", "source": "briefing:national_status",
                 "type": "briefing", "text":
                 f"PM Surya Ghar national status. Across {df['state'].nunique()} "
                 f"states and UTs, {inst:,.0f} rooftop installations have been "
                 f"completed from {apps:,.0f} applications, a national application-"
                 f"to-installation conversion ratio of {inst/apps:.1%}. Total central "
                 f"subsidy released is about Rs {df['subsidy_released_cr'].sum():,.0f} "
                 f"crore for {df['capacity_mw'].sum():,.0f} MW of rooftop capacity. "
                 f"Installations are led by {', '.join(top_inst)}. Conversion is "
                 f"highest in {', '.join(top_conv)} and lowest in {', '.join(low_conv)}."})
    docs.append({"id": "brief::findings", "source": "briefing:key_findings",
                 "type": "briefing", "text":
                 "Key analytical findings. Conversion varies enormously across "
                 "states (roughly 2% to 93%). A leave-one-state-out model shows "
                 "implementation-quality signals (registration-to-application ratio, "
                 "applications per lakh population, potential utilisation) predict "
                 "conversion better than the amount of subsidy released — the "
                 "'subsidy paradox'. This means disbursement speed and DISCOM "
                 "execution matter more than headline subsidy size. Tariff has a "
                 "weak association with conversion at the state level."})
    # tariff + potential briefing
    hi_tar = df.nlargest(5, "tariff_2kw_2025")[["state", "tariff_2kw_2025"]]
    hi_pot = df.nlargest(5, "solar_potential_gw")[["state", "solar_potential_gw"]]
    docs.append({"id": "brief::tariff_potential", "source": "briefing:tariff_and_potential",
                 "type": "briefing", "text":
                 "Tariff and solar-potential context. Highest residential 2 kW "
                 "tariffs (Rs/kWh, 2025): " +
                 "; ".join(f"{r['state']} {r['tariff_2kw_2025']:.2f}" for _, r in hi_tar.iterrows()) +
                 ". Largest solar potential (GW): " +
                 "; ".join(f"{r['state']} {r['solar_potential_gw']:.0f}" for _, r in hi_pot.iterrows()) +
                 ". Rooftop economics improve where tariffs and irradiance are high "
                 "and subsidy shortens payback."})
    # per-region briefings
    if "region" in df:
        for reg, g in df.dropna(subset=["region"]).groupby("region"):
            docs.append({"id": f"brief::region_{reg}",
                         "source": f"briefing:region_{reg}", "type": "briefing",
                         "text": f"Regional summary ({reg}). States: "
                         f"{', '.join(g['state'])}. Mean conversion "
                         f"{g['conversion_ratio'].mean():.1%}, total installations "
                         f"{g['installations'].sum():,.0f}, total subsidy Rs "
                         f"{g['subsidy_released_cr'].sum():,.0f} crore."})
    return docs


def _synthesis_docs():
    f = ROOT / "docs" / "SuryaIQ_Cross-Section_Synthesis.xlsx"
    if not f.exists():
        return []
    docs = []
    try:
        xls = pd.read_excel(f, sheet_name=None, header=None)
    except Exception:
        return []
    for sheet, d in xls.items():
        text = " ".join(str(v) for v in d.values.ravel() if pd.notna(v))
        docs.append({"id": f"lit::{sheet}", "source": f"literature:{sheet}",
                     "type": "literature", "text": text[:6000]})
    return docs


class Retriever:
    def __init__(self):
        self.docs = (_briefing_docs() + _state_factsheets()
                     + _synthesis_docs() + _markdown_docs())
        self.bm25 = BM25Okapi([_tok(d["text"]) for d in self.docs])

    def search(self, query, k=4, type_filter=None):
        scores = self.bm25.get_scores(_tok(query))
        ranked = sorted(zip(scores, self.docs), key=lambda t: -t[0])
        out = []
        for sc, d in ranked:
            if type_filter and d["type"] != type_filter:
                continue
            if sc <= 0:
                continue
            out.append({"source": d["source"], "type": d["type"],
                        "score": round(float(sc), 2), "text": d["text"][:500]})
            if len(out) >= k:
                break
        return out


_R = None
def get_retriever():
    global _R
    if _R is None:
        _R = Retriever()
    return _R


if __name__ == "__main__":
    r = get_retriever()
    print(f"corpus docs: {len(r.docs)}")
    for q in ["Why is Rajasthan performing better than Odisha?",
              "subsidy paradox implementation quality",
              "Maharashtra conversion ratio"]:
        print(f"\nQ: {q}")
        for hit in r.search(q, k=3):
            print(f"  [{hit['score']}] {hit['source']}: {hit['text'][:90]}...")
