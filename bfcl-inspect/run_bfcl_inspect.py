#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run BFCL multi-turn (V3) tool-calling evaluation via Inspect AI against an
*already running* OpenAI-compatible server.

Uses the `inspect_evals/bfcl` task from UKGovernmentBEIS/inspect_evals, which
implements BFCL's stateful multi-turn categories (GorillaFileSystem, TradingBot,
VehicleControl, ... backends with state-based + response-based checking) on top
of the Inspect AI eval framework.

Requires: uv pip install "inspect-ai>=0.3.258" "inspect-evals[bfcl]" openai

Examples:
    # Server on localhost:8000, model auto-detected from /v1/models,
    # default categories: the four BFCL v3 multi-turn categories
    python run_bfcl_inspect.py

    # Explicit model / categories / remote server
    python run_bfcl_inspect.py \
        --base-url http://my-host:8000/v1 \
        --model openai/gpt-oss-120b \
        --test-category multi_turn_base multi_turn_miss_func \
        --output-dir ./bfcl-inspect-results

    # Smoke test: only the first 5 entries of each test category
    python run_bfcl_inspect.py --num-prompts 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

# BFCL v3 "multi_turn" group, matching bfcl_eval's own multi_turn collection.
# (multi_turn_composite exists in the dataset but is not part of the
# leaderboard's multi_turn group; request it explicitly if you want it.)
MULTI_TURN_CATEGORIES = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]

# Group aliases accepted by --test-category, mirroring run_bfcl_eval.py's
# category-group ergonomics.
CATEGORY_GROUPS = {
    "multi_turn": MULTI_TURN_CATEGORIES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible base URL of the running server "
        "(default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to evaluate. Defaults to the first model reported by "
        "the server's /v1/models endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "dummy"),
        help="API key sent to the server (default: $OPENAI_API_KEY or 'dummy')",
    )
    parser.add_argument(
        "--test-category",
        nargs="+",
        default=["multi_turn"],
        help="BFCL test categories (default: multi_turn, which expands to "
        f"{', '.join(MULTI_TURN_CATEGORIES)}). Accepts inspect_evals/bfcl "
        "category names, the 'multi_turn' group alias, multiple "
        "space-separated values, or a single comma-separated string.",
    )
    parser.add_argument(
        "--num-prompts",
        "-n",
        type=int,
        default=None,
        help="Run only the first N test entries of each category instead of "
        "the full set.",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=32,
        help="Maximum concurrent connections to the server (default: 32)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max completion tokens per request (default: server default)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for Inspect eval logs and the results summary "
        "(default: cwd)",
    )
    parser.add_argument(
        "--display",
        choices=["full", "conversation", "rich", "plain", "log", "none"],
        default="plain",
        help="Inspect progress display (default: plain)",
    )
    args = parser.parse_args()

    # Allow --test-category "a, b" in addition to --test-category a b,
    # then expand group aliases.
    categories: list[str] = []
    for chunk in args.test_category:
        for name in (c.strip() for c in chunk.split(",")):
            if not name:
                continue
            categories.extend(CATEGORY_GROUPS.get(name, [name]))
    # De-duplicate, preserving order.
    args.test_category = list(dict.fromkeys(categories))
    args.base_url = args.base_url.rstrip("/")
    if args.num_prompts is not None and args.num_prompts < 1:
        parser.error("--num-prompts must be >= 1")
    return args


def detect_model(base_url: str, api_key: str) -> str:
    """Ask the running server which model it is serving."""
    request = urllib.request.Request(
        f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        sys.exit(
            f"ERROR: could not reach {base_url}/models ({exc}).\n"
            "Is the server running? Pass --base-url / --model explicitly."
        )

    models = payload.get("data") or []
    if not models:
        sys.exit(f"ERROR: {base_url}/models returned no models; pass --model.")
    return models[0]["id"]


def validate_categories(categories: list[str]) -> None:
    from inspect_evals.bfcl.utils.task_categories import ALL_CATEGORIES

    unknown = [c for c in categories if c not in ALL_CATEGORIES]
    if unknown:
        sys.exit(
            f"ERROR: unknown test categories: {', '.join(unknown)}.\n"
            f"Known categories: {', '.join(sorted(ALL_CATEGORIES))}"
        )


def summarize(logs) -> tuple[dict, bool]:
    """Collapse Inspect eval logs into {category: {metric: value}}."""
    summary: dict[str, dict] = {}
    ok = True
    for log in logs:
        if log.status != "success":
            ok = False
        entry: dict = {"status": log.status}
        if log.results is not None:
            entry["total_samples"] = log.results.total_samples
            entry["completed_samples"] = log.results.completed_samples
            for score in log.results.scores:
                for name, metric in score.metrics.items():
                    entry[name] = metric.value
        for category in log.eval.task_args.get("categories") or ["bfcl"]:
            summary[category] = entry
    return summary, ok


def main() -> int:
    args = parse_args()

    model = args.model or detect_model(args.base_url, args.api_key)
    output_dir = os.path.abspath(args.output_dir or os.getcwd())
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 52)
    print("BFCL Multi-Turn Evaluation (Inspect AI)")
    print("=" * 52)
    print(f"Base URL:         {args.base_url}")
    print(f"Model:            {model}")
    print(f"Test category:    {', '.join(args.test_category)}")
    if args.num_prompts is not None:
        print(f"Num prompts:      {args.num_prompts} per category")
    print(f"Max connections:  {args.max_connections}")
    print(f"Temperature:      {args.temperature}")
    print(f"Output dir:       {output_dir}")
    print("=" * 52)

    try:
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import get_model
        from inspect_evals.bfcl import bfcl
    except ImportError as exc:
        sys.exit(
            f"ERROR: {exc}\n"
            "Install with: uv pip install "
            '"inspect-ai>=0.3.258" "inspect-evals[bfcl]" openai'
        )

    validate_categories(args.test_category)

    # openai-api/<service>/<model> is Inspect's generic OpenAI-compatible
    # provider; the "local/" service prefix is stripped from requests, so the
    # server sees the bare model name.
    inspect_model = get_model(
        f"openai-api/local/{model}",
        base_url=args.base_url,
        api_key=args.api_key,
    )

    # One task per category keeps --num-prompts per-category (Inspect's
    # `limit` applies per task) and yields a per-category log file.
    tasks = [bfcl(categories=[category]) for category in args.test_category]

    logs = inspect_eval(
        tasks,
        model=inspect_model,
        limit=args.num_prompts,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_connections=args.max_connections,
        log_dir=log_dir,
        display=args.display,
    )

    summary, ok = summarize(logs)
    summary_path = os.path.join(output_dir, "bfcl_inspect_summary.json")
    with open(summary_path, "w") as summary_file:
        json.dump({"model": model, "results": summary}, summary_file, indent=2)

    print()
    print("=== BFCL multi-turn (Inspect AI) results ===")
    for category, entry in summary.items():
        accuracy = entry.get("accuracy")
        shown = f"accuracy={accuracy:.4f}" if accuracy is not None else entry["status"]
        print(f"  {category:28s} {shown}")
    print(f"Summary: {summary_path}")
    print(f"Logs:    {log_dir}")

    if not ok:
        print("ERROR: one or more eval tasks did not complete successfully")
        return 1
    print("=== BFCL evaluation completed successfully ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
