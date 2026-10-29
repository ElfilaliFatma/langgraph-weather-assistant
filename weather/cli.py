"""CLI for the weather assistant.

    python -m weather.cli "weather in Tunis"
    python -m weather.cli --offline "forecast for Sousse this weekend"
    python -m weather.cli --trace "compare Paris and London"
    python -m weather.cli            # interactive, keeps one thread
"""

from __future__ import annotations

import argparse
import sys

from .config import Settings
from .graph import WeatherAssistant, build_compiled_graph


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="weather", description="Weather assistant.")
    parser.add_argument("question", nargs="?", help="Natural language request.")
    parser.add_argument("--offline", action="store_true", help="No LLM (rules only).")
    parser.add_argument("--trace", action="store_true", help="Print the graph path.")
    parser.add_argument("--thread", default="cli", help="Conversation/thread id.")
    parser.add_argument("--json", action="store_true", help="Print the final state.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings.from_env()
    graph = build_compiled_graph(settings, with_llm=not args.offline)
    assistant = WeatherAssistant(graph, settings)

    def handle(text: str) -> int:
        run = assistant.ask(text, args.thread)
        if args.json:
            import json

            print(json.dumps(run.state, ensure_ascii=False, indent=2, default=str))
            return 0 if run.status != "error" else 1
        if args.trace:
            print("graph path:")
            for step in run.trail:
                print(f"  · {step}")
            for call in run.state.get("tool_calls", []):
                print(f"  ↳ {call}")
            print()
        print(run.answer)
        return 0 if run.status != "error" else 1

    if args.question:
        return handle(args.question)

    print(f"Weather assistant (thread={args.thread}, offline={args.offline})")
    print("Type 'exit' to quit.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            return 0
        try:
            handle(text)
        except Exception as exc:
            print(f"[error] {exc}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
