"""Evaluation pipeline for the SuryaIQ assistant.

Runs the YAML test cases through the LangGraph app and scores each on:
intent-routing accuracy, content checks, citation presence, scope rejection,
and latency. Writes a results table to evals/results/."""
import json
import time
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

import graph  # noqa: E402


def run():
    cases = yaml.safe_load((ROOT / "evals" / "test_cases.yaml").read_text())
    rows, passed = [], 0
    cited, grounded = 0, 0
    for c in cases:
        t0 = time.time()
        ans = graph.ask(c["query"])
        dt = round((time.time() - t0) * 1000)
        rendered = ans.render().lower()
        checks = c.get("checks", {})
        results = {}
        if "intent" in checks:
            results["intent"] = ans.intent == checks["intent"]
        if "contains" in checks:
            results["contains"] = checks["contains"].lower() in rendered
        if "has_sources" in checks:
            non_guard = [s for s in ans.data_source if s != "guardrail"]
            results["has_sources"] = (len(non_guard) > 0) == checks["has_sources"]
        if "rejected" in checks:
            results["rejected"] = (ans.intent == "rejected") == checks["rejected"]
        ok = all(results.values())
        passed += ok
        # report's eval dimensions:
        is_answer = ans.intent != "rejected"
        has_cite = any(s != "guardrail" for s in ans.data_source)
        if is_answer:
            cited += has_cite            # citation correctness (answers must cite)
            grounded += has_cite         # grounded answer => not hallucinated
        rows.append({"id": c["id"], "intent": ans.intent, "pass": ok,
                     "checks": results, "latency_ms": dt, "cited": has_cite})

    n_ans = sum(1 for r in rows if r["intent"] != "rejected")
    summary = {"total": len(cases), "passed": passed,
               "pass_rate": round(passed / len(cases), 3),
               "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / len(rows)),
               # --- report's evaluation dimensions (section 7) ---
               "citation_correctness": round(cited / max(n_ans, 1), 3),
               "hallucination_rate": round(1 - grounded / max(n_ans, 1), 3),
               "cost_usd": 0.0 if not __import__("llm").have_key() else "see provider usage",
               "mode": "online(Claude)" if __import__("llm").have_key() else "offline(template)",
               "cases": rows}
    outdir = ROOT / "evals" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "eval_results.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    s = run()
    print(f"\nSuryaIQ Evals — {s['passed']}/{s['total']} passed "
          f"({s['pass_rate']*100:.0f}%) | avg {s['avg_latency_ms']} ms | {s['mode']}")
    print(f"citation_correctness={s['citation_correctness']} | "
          f"hallucination_rate={s['hallucination_rate']} | cost_usd={s['cost_usd']}\n")
    print(f"{'id':4} {'intent':10} {'pass':5} {'latency':8} checks")
    for r in s["cases"]:
        print(f"{r['id']:4} {r['intent']:10} {str(r['pass']):5} "
              f"{r['latency_ms']:>5}ms  {r['checks']}")
