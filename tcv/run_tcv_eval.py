#!/usr/bin/env python3
"""Run Tool-Call Verifier (TCV) suites against an *already running*
OpenAI-compatible server, such as an llm-d endpoint serving GLM.

Never starts or stops a server; point it at one you deployed yourself.

Suites (select with --suites):
    behavior     Curated tool-call behavior cases from cases/*.jsonl
                 (parsing edge cases, parallel calls, multi-turn round trips)
    tool_choice  tool_choice contract: auto / none / required / named
    schema       Tool-call arguments fuzzed against walle JSON Schemas,
                 streaming and non-streaming

Examples:
    # GLM behind an llm-d gateway, all suites (default), GLM profile (default)
    python run_tcv_eval.py --base-url http://llm-d-gateway:8000/v1

    # Behavior cases only, thinking explicitly off
    python run_tcv_eval.py --suites behavior --thinking off

    # Schema fuzzing capped at 50 cases for a quick signal
    python run_tcv_eval.py --suites schema --max-cases 50

    # A different model family
    python run_tcv_eval.py --profile qwen3 --model Qwen3-32B
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

TCV_DIR = os.path.dirname(os.path.abspath(__file__))

SUITES = ("behavior", "tool_choice", "schema")
SUITE_PATHS = {
    "behavior": "tests/behavior",
    "tool_choice": "tests/tool_choice",
    "schema": "tests/schema",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "TCV_BASE_URL",
            os.environ.get("BASE_URL", "http://localhost:8000/v1"),
        ),
        help="OpenAI-compatible base URL of the running server "
        "(default: $TCV_BASE_URL / $BASE_URL / http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TCV_API_KEY", "dummy"),
        help="API key for the server (default: $TCV_API_KEY or 'dummy')",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME") or None,
        help="Model name to evaluate. Defaults to $MODEL_NAME, else the first "
        "model reported by the server's /v1/models endpoint.",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("TCV_PROFILE", "glm"),
        help="Model-family profile: glm, qwen3, kimi, deepseek, generic "
        "(default: $TCV_PROFILE or glm)",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=list(SUITES),
        help=f"Suites to run (default: all). Choices: {', '.join(SUITES)}. "
        "Accepts space- or comma-separated values, or 'all'.",
    )
    parser.add_argument(
        "--thinking",
        choices=["default", "on", "off"],
        default=os.environ.get("TCV_THINKING", "default"),
        help="Thinking mode: 'default' sends nothing (server/template "
        "default), 'on'/'off' send the profile's chat_template_kwargs "
        "toggle (default: default)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="pytest-xdist workers for the behavior and schema suites "
        "(default: 4)",
    )
    parser.add_argument(
        "--reruns",
        type=int,
        default=2,
        help="Retries for flaky cases (default: 2)",
    )
    parser.add_argument(
        "--reruns-delay",
        type=int,
        default=2,
        help="Seconds between retries (default: 2)",
    )
    parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated tag filter for behavior cases (default: all)",
    )
    parser.add_argument(
        "--case-filter",
        default="",
        help="Substring filter on behavior case ids (default: all)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Cap on schema-suite cases (default: all)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Default max output tokens per request (default: 2048)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for reports, junit XML, and summary.json (default: cwd)",
    )
    args = parser.parse_args()

    suites: list[str] = []
    for chunk in args.suites:
        suites.extend(s.strip() for s in chunk.split(",") if s.strip())
    if "all" in suites:
        suites = list(SUITES)
    unknown = [s for s in suites if s not in SUITES]
    if unknown:
        parser.error(
            f"unknown suites: {', '.join(unknown)} (choose from {', '.join(SUITES)})"
        )
    args.suites = suites
    args.base_url = args.base_url.rstrip("/")
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


def pytest_command(suite: str, args: argparse.Namespace, out: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        SUITE_PATHS[suite],
        "--base-url",
        args.base_url,
        "--api-key",
        args.api_key,
        "--model",
        args.model,
        "--profile",
        args.profile,
        "--thinking",
        args.thinking,
        "--max-tokens",
        str(args.max_tokens),
        f"--junitxml={os.path.join(out, f'junit-{suite}.xml')}",
        "-ra",
        "-v",
    ]
    if suite in ("behavior", "schema"):
        cmd += [
            "-n",
            str(args.num_workers),
            "--reruns",
            str(args.reruns),
            "--reruns-delay",
            str(args.reruns_delay),
        ]
    if suite == "behavior":
        cmd.append(f"--behavior-report={os.path.join(out, 'behavior-report.json')}")
        if args.tags:
            cmd += ["--tags", args.tags]
        if args.case_filter:
            cmd += ["--case-filter", args.case_filter]
    elif suite == "schema":
        cmd.append(f"--schema-report={os.path.join(out, 'schema-report.json')}")
        if args.max_cases is not None:
            cmd += ["--max-cases", str(args.max_cases)]
    return cmd


def main() -> int:
    args = parse_args()

    model = args.model or detect_model(args.base_url, args.api_key)
    args.model = model
    output_dir = os.path.abspath(args.output_dir or os.getcwd())
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Tool-Call Verifier — tool-calling quality evaluation")
    print("=" * 60)
    print(f"Base URL:   {args.base_url}")
    print(f"Model:      {model}")
    print(f"Profile:    {args.profile} (thinking={args.thinking})")
    print(f"Suites:     {', '.join(args.suites)}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    results: dict[str, int] = {}
    for suite in args.suites:
        cmd = pytest_command(suite, args, output_dir)
        print(f"\n=== [{suite}] {' '.join(cmd)} ===\n", flush=True)
        proc = subprocess.run(cmd, cwd=TCV_DIR)
        results[suite] = proc.returncode
        status = "PASS" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
        print(f"\n=== [{suite}] {status} ===", flush=True)

    summary = {
        "base_url": args.base_url,
        "model": model,
        "profile": args.profile,
        "thinking": args.thinking,
        "suites": {
            suite: {"exit_code": code, "passed": code == 0}
            for suite, code in results.items()
        },
        "passed": all(code == 0 for code in results.values()),
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Suite results")
    print("=" * 60)
    for suite, code in results.items():
        print(f"  {suite:<12} {'PASS' if code == 0 else f'FAIL (exit {code})'}")
    print(f"\nSummary: {summary_path}")
    print(f"Reports: {output_dir}")

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
