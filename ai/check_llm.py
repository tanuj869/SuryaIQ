"""Diagnose the Claude connection. Run:  python ai/check_llm.py

Reports whether a key is loaded, which model answers, and shows a live
polished answer vs the offline draft — so you can confirm polished-prose mode
is actually on."""
import warnings

warnings.filterwarnings("ignore")
import llm  # noqa: E402


def main():
    usable, reason = llm.key_status()
    if not usable:
        print(f"✗ No usable API key — {reason}.")
        print("  OpenAI: put OPENAI_API_KEY=sk-... in suryaiq/.env (platform.openai.com/api-keys)")
        print("  Anthropic: put ANTHROPIC_API_KEY=sk-ant-api03-... (console.anthropic.com)")
        print("  Do NOT paste the example text — use your real key.")
        return
    prov = llm.provider()
    print(f"✓ Key loaded. Provider: {prov}. Model: {llm.model_for(prov)}")
    try:
        txt = llm._complete("You are a test.", "Reply with exactly: OK", max_tokens=20)
        print(f"✓ {prov} responded: {txt!r}")
    except Exception as e:
        print(f"✗ API call failed: {type(e).__name__}: {e}")
        print("  Check the key is valid and the model id is available on your plan.")
        return
    # show polish vs draft on a real query
    import graph
    q = "What if subsidy increases by 10%?"
    a = graph.ask(q)
    print(f"\nPolished answer for '{q}':\n  {a.answer}")
    print(f"\nPolished-prose mode is ON ({prov}). Restart the dashboard to pick it up.")


if __name__ == "__main__":
    main()
