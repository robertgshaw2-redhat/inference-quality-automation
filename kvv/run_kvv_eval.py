#!/usr/bin/env python3
"""Run Kimi-Vendor-Verifier (KVV) suites against an *already running*
OpenAI-compatible server, such as an llm-d endpoint.

Wraps the vendored MoonshotAI/kimi-vendor-verifier checkout (see vendor/):
never starts or stops a server; point it at one you deployed yourself.

Suites (select with --suites):
    Pytest verifiers (tool-calling / API-contract focused):
        params        API parameter constraint pre-flight validation
        tool_schema   Tool-call arguments validated against walle JSON schemas
        k3_features   K3 feature contract (dynamic tools, response_format,
                      tool_choice, thinking effort)
        prompt_tokens usage.prompt_tokens accuracy vs expected constants
    Inspect-ai benchmarks (require benchmark-specific model capabilities,
    and network access to download datasets on first run):
        ocrbench      OCR text recognition (vision)
        mmmu          MMMU Pro Vision (vision)
        aime2025      AIME 2025 math reasoning (32 epochs by default)

Examples:
    # llm-d endpoint on the cluster, tool-calling verifier suites (default)
    python run_kvv_eval.py --base-url http://llm-d-gateway:8000/v1

    # Just the tool-call schema suite, thinking on, capped at 50 cases
    python run_kvv_eval.py --suites tool_schema --thinking --max-cases 50

    # OCRBench sanity check then MMMU
    python run_kvv_eval.py --suites ocrbench mmmu --thinking
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")

PYTEST_SUITES = ("params", "tool_schema", "k3_features", "prompt_tokens")
BENCHMARK_SUITES = ("ocrbench", "mmmu", "aime2025")

# Recommended max output tokens per benchmark (vendor README).
BENCHMARK_MAX_TOKENS = {
    "ocrbench": 16384,
    "mmmu": 98304,
    "aime2025": 98304,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "KIMI_BASE_URL",
            os.environ.get("BASE_URL", "http://localhost:8000/v1"),
        ),
        help="OpenAI-compatible base URL of the running server "
        "(default: $KIMI_BASE_URL / $BASE_URL / http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("KIMI_API_KEY", "dummy"),
        help="API key for the server (default: $KIMI_API_KEY or 'dummy')",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME") or None,
        help="Model name to evaluate. Defaults to $MODEL_NAME, else the first "
        "model reported by the server's /v1/models endpoint.",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=list(PYTEST_SUITES),
        help="Suites to run (default: the four pytest verifier suites). "
        f"Choices: {', '.join(PYTEST_SUITES + BENCHMARK_SUITES)}. Accepts "
        "space- or comma-separated values, or 'all'.",
    )
    parser.add_argument(
        "--think-mode",
        choices=["opensource", "kimi", "none"],
        default=os.environ.get("THINK_MODE", "opensource"),
        help="Thinking parameter format: 'opensource' for vLLM/SGLang-style "
        "chat_template_kwargs (llm-d), 'kimi' for the Moonshot SaaS API, "
        "'none' to send no thinking params (default: $THINK_MODE or "
        "opensource)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode for tool-call schema requests and "
        "benchmarks (default: off)",
    )
    parser.add_argument(
        "--thinking-effort",
        choices=["low", "high", "max"],
        default=None,
        help="K3 reasoning effort; only sent when --thinking is set",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="pytest-xdist workers for the tool_schema suite (default: 4)",
    )
    parser.add_argument(
        "--reruns",
        type=int,
        default=3,
        help="Retries for flaky tool_schema cases (default: 3)",
    )
    parser.add_argument(
        "--reruns-delay",
        type=int,
        default=2,
        help="Seconds between tool_schema retries (default: 2)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Cap on tool_schema cases (default: all)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max output tokens per tool_schema response (default: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Benchmark sampling temperature (default: server/model default; "
        "vendor README recommends 1.0 for thinking)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Benchmark top-p (vendor README recommends 0.95)",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=None,
        help="Benchmark max concurrent connections (default: per benchmark)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Benchmark sampling epochs (default: per benchmark; aime2025 "
        "defaults to 32)",
    )
    parser.add_argument(
        "--client-timeout",
        type=int,
        default=86400,
        help="Benchmark HTTP timeout in seconds (default: 86400)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for reports, logs, and summary.json (default: cwd)",
    )
    args = parser.parse_args()

    suites: list[str] = []
    for chunk in args.suites:
        suites.extend(s.strip() for s in chunk.split(",") if s.strip())
    if "all" in suites:
        suites = list(PYTEST_SUITES + BENCHMARK_SUITES)
    unknown = [s for s in suites if s not in PYTEST_SUITES + BENCHMARK_SUITES]
    if unknown:
        parser.error(
            f"unknown suites: {', '.join(unknown)} "
            f"(choose from {', '.join(PYTEST_SUITES + BENCHMARK_SUITES)})"
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
        "--base-url",
        args.base_url,
        "--api-key",
        args.api_key,
        "--smoke-model",
        args.model,
        "--think-mode",
        args.think_mode,
        f"--junitxml={os.path.join(out, f'junit-{suite}.xml')}",
        "-ra",
        "-v",
    ]
    if suite == "params":
        cmd.append("tests/params")
    elif suite == "k3_features":
        cmd.append("tests/k3_features")
    elif suite == "prompt_tokens":
        cmd.append("tests/prompt_tokens")
    elif suite == "tool_schema":
        cmd += [
            "tests/tool_call_json_schema",
            "-n",
            str(args.num_workers),
            "--reruns",
            str(args.reruns),
            "--reruns-delay",
            str(args.reruns_delay),
            "--max-tokens",
            str(args.max_tokens),
            f"--tool-json-report={os.path.join(out, 'tool-call-schema-report.json')}",
        ]
        if args.thinking:
            cmd.append("--thinking")
        if args.max_cases is not None:
            cmd += ["--max-cases", str(args.max_cases)]
    return cmd


def benchmark_command(suite: str, args: argparse.Namespace) -> list[str]:
    provider = "opensource" if args.think_mode == "opensource" else "kimi"
    cmd = [
        sys.executable,
        "eval.py",
        suite,
        "--model",
        f"{provider}/{args.model}",
        "--max-tokens",
        str(BENCHMARK_MAX_TOKENS[suite]),
        "--think-mode",
        args.think_mode,
        "--client-timeout",
        str(args.client_timeout),
        "--stream",
    ]
    if args.thinking:
        cmd.append("--thinking")
    if args.thinking_effort:
        cmd += ["--thinking-effort", args.thinking_effort]
    if args.temperature is not None:
        cmd += ["--temperature", str(args.temperature)]
    if args.top_p is not None:
        cmd += ["--top-p", str(args.top_p)]
    if args.max_connections is not None:
        cmd += ["--max-connections", str(args.max_connections)]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    return cmd


def main() -> int:
    args = parse_args()

    model = args.model or detect_model(args.base_url, args.api_key)
    args.model = model
    output_dir = os.path.abspath(args.output_dir or os.getcwd())
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["KIMI_BASE_URL"] = args.base_url
    env["KIMI_API_KEY"] = args.api_key
    env["MODEL_NAME"] = model
    env["THINK_MODE"] = args.think_mode
    # inspect-ai benchmark logs go to the results volume
    env.setdefault("INSPECT_LOG_DIR", os.path.join(output_dir, "logs"))

    print("=" * 60)
    print("Kimi Vendor Verifier — inference quality evaluation")
    print("=" * 60)
    print(f"Base URL:    {args.base_url}")
    print(f"Model:       {model}")
    print(f"Think mode:  {args.think_mode} (thinking={'on' if args.thinking else 'off'}"
          + (f", effort={args.thinking_effort}" if args.thinking_effort else "") + ")")
    print(f"Suites:      {', '.join(args.suites)}")
    print(f"Output dir:  {output_dir}")
    print("=" * 60)

    results: dict[str, int] = {}
    for suite in args.suites:
        if suite in PYTEST_SUITES:
            cmd = pytest_command(suite, args, output_dir)
        else:
            cmd = benchmark_command(suite, args)
        print(f"\n=== [{suite}] {' '.join(cmd)} ===\n", flush=True)
        proc = subprocess.run(cmd, cwd=VENDOR_DIR, env=env)
        results[suite] = proc.returncode
        status = "PASS" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
        print(f"\n=== [{suite}] {status} ===", flush=True)

    summary = {
        "base_url": args.base_url,
        "model": model,
        "think_mode": args.think_mode,
        "thinking": args.thinking,
        "thinking_effort": args.thinking_effort,
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
        print(f"  {suite:<14} {'PASS' if code == 0 else f'FAIL (exit {code})'}")
    print(f"\nSummary: {summary_path}")
    print(f"Reports: {output_dir}")

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
