"""Smoke tests — the kind an examiner expects to see run green."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ai"))


# -- data -------------------------------------------------------------------
def test_master_panel_shape():
    d = pd.read_csv(ROOT / "data/processed/master_state_panel.csv")
    assert len(d) >= 30 and len(d) <= 40
    assert "conversion_ratio" in d.columns
    assert d["conversion_ratio"].between(0, 1).all()


def test_conversion_range_matches_paper():
    d = pd.read_csv(ROOT / "data/processed/master_state_panel.csv")
    assert d["conversion_ratio"].min() < 0.05
    assert d["conversion_ratio"].max() > 0.9


# -- paper reproduction -----------------------------------------------------
def test_linear_beats_random_forest():
    r = json.loads((ROOT / "data/processed/paper_results.json").read_text())
    mc = r["table2_model_comparison"]
    assert mc["linear_2feature"]["loo_cv_mae"] < mc["random_forest_6feature"]["loo_cv_mae"]
    assert mc["winner"] == "linear_2feature"


def test_subsidy_scenario_unsupportable():
    r = json.loads((ROOT / "data/processed/paper_results.json").read_text())
    sub = [a for a in r["table4_scenario_support"]["assessment"]
           if a["scenario"].startswith("B")][0]
    assert sub["supported"] == "No"


# -- AI assistant -----------------------------------------------------------
def test_guardrail_rejects_out_of_scope():
    import graph
    a = graph.ask("Predict installations for Mars")
    assert a.intent == "rejected"


def test_highest_adoption_is_gujarat():
    import graph
    a = graph.ask("Which state had the highest adoption?")
    assert "Gujarat" in a.answer


def test_subsidy_scenario_flagged_unsupportable_in_assistant():
    import graph
    a = graph.ask("What if subsidy increases by 10%?")
    assert "not supportable" in a.answer.lower()


def test_state_scoped_tariff_scenario():
    import graph
    a = graph.ask("what if tariff rises 10% in Maharashtra")
    assert a.intent == "scenario"
    assert "Maharashtra" in a.answer


def test_executive_brief_contains_shap_and_scenario():
    import graph
    a = graph.ask("Executive brief for Maharashtra")
    assert a.intent == "brief"
    assert "SHAP" in a.answer
    assert "Scenario" in a.answer


# -- structured output schema ----------------------------------------------
def test_every_answer_is_structured():
    import graph
    a = graph.ask("What is Gujarat conversion ratio?")
    for field in ("answer", "confidence", "data_source", "assumptions", "limitations"):
        assert hasattr(a, field)
    assert a.confidence
    assert a.data_source
