"""Interactive SuryaIQ assistant (terminal).

Usage:
    python ai/cli.py                 # interactive
    python ai/cli.py "your question" # one-shot

Set ANTHROPIC_API_KEY to get Claude-polished answers; otherwise runs offline
with deterministic, fully-cited template answers.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import graph  # noqa: E402
import llm    # noqa: E402

BANNER = (
    "SuryaIQ — PM Surya Ghar adoption assistant  "
    f"[{'Claude' if llm.have_key() else 'offline'} mode]\n"
    "Ask about conversion, forecasts, scenarios, or state comparisons. "
    "Type 'quit' to exit.\n" + "-" * 64
)


def answer(q):
    print(graph.ask(q).render())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        answer(" ".join(sys.argv[1:]))
    else:
        print(BANNER)
        while True:
            try:
                q = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                print()
                answer(q)
