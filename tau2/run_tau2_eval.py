#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run tau2-bench agentic tool-calling evaluation against an *already running*
OpenAI-compatible server.

tau2-bench (https://github.com/sierra-research/tau2-bench) drops the model
being evaluated into a customer-service agent loop: it must chat with a
simulated user, follow a domain policy, and call domain tools (flight
changes, order returns, ...) against a mock database. A task passes only if
the final database state and the info communicated to the user match the
task's goal, so it exercises multi-turn tool calling end to end.

Both the agent under test and the user simulator default to the model served
at --base-url, so no external API keys are needed. Point --user-llm at any
LiteLLM identifier (e.g. gpt-4.1) to use a stronger user simulator instead.

Examples:
    # Server on localhost:8000, model auto-detected from /v1/models,
    # full retail domain (114 tasks)
    python run_tau2_eval.py

    # Smoke test: 5 airline tasks, 1 trial
    python run_tau2_eval.py --domain airline --num-tasks 5

    # More reliability signal: pass^k over 4 trials of 20 tasks
    python run_tau2_eval.py --num-tasks 20 --num-trials 4

Any unrecognized arguments are forwarded verbatim to `tau2 run`
(see `tau2 run --help`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
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
        "--domain",
        default="retail",
        help="tau2 domain: retail (114 tasks), airline (50), telecom, mock "
        "(default: retail)",
    )
    parser.add_argument(
        "--num-tasks",
        "-n",
        type=int,
        default=None,
        help="Run only the first N tasks of the domain (default: all)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Trials per task; >1 enables pass^k metrics (default: 1)",
    )
    parser.add_argument(
        "--user-llm",
        default=None,
        help="LiteLLM model for the user simulator. Default: the same local "
        "model being evaluated. Anything LiteLLM supports works "
        "(e.g. gpt-4.1 with a real OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the agent under test (default: 0.0)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Concurrent simulations (default: 8)",
    )
    parser.add_argument(
        "--save-to",
        default=None,
        help="Run name; results land in <output-dir>/<save-to>/results.json. "
        "Re-using a name resumes that run. Default: tau2 picks "
        "<timestamp>_<domain>_llm_agent_user_simulator.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for simulation results (default: tau2's data dir)",
    )
    return parser.parse_known_args()


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


def link_output_dir(output_dir: str) -> None:
    """Make tau2 write simulation results into output_dir.

    tau2 hardcodes $TAU2_DATA_DIR/simulations as its output root, and
    $TAU2_DATA_DIR must stay pointed at the tau2-bench checkout because the
    domain task data lives there too. Symlinking the simulations dir is the
    supported-layout way to redirect results to the mounted volume.
    """
    data_dir = os.environ.get("TAU2_DATA_DIR")
    if not data_dir:
        sys.exit("ERROR: --output-dir requires TAU2_DATA_DIR to be set.")
    simulations = Path(data_dir) / "simulations"
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    if simulations.is_symlink():
        simulations.unlink()
    elif simulations.exists():
        sys.exit(
            f"ERROR: {simulations} already exists and is not a symlink; "
            "refusing to redirect results."
        )
    simulations.symlink_to(target)


def main() -> int:
    args, extra_args = parse_args()
    args.base_url = args.base_url.rstrip("/")

    model = args.model or detect_model(args.base_url, args.api_key)
    # LiteLLM's openai/ prefix routes to any OpenAI-compatible endpoint;
    # the base URL and key are picked up from the environment below.
    agent_llm = model if model.startswith("openai/") else f"openai/{model}"
    user_llm = args.user_llm or agent_llm

    os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["OPENAI_BASE_URL"] = args.base_url
    os.environ["OPENAI_API_BASE"] = args.base_url

    if args.output_dir:
        link_output_dir(args.output_dir)

    print("=" * 44)
    print("tau2-bench Agentic Tool Calling Evaluation")
    print("=" * 44)
    print(f"Base URL:        {args.base_url}")
    print(f"Agent LLM:       {agent_llm}")
    print(f"User sim LLM:    {user_llm}")
    print(f"Domain:          {args.domain}")
    print(f"Num tasks:       {args.num_tasks if args.num_tasks else 'all'}")
    print(f"Num trials:      {args.num_trials}")
    print(f"Temperature:     {args.temperature}")
    print(f"Max concurrency: {args.max_concurrency}")
    if extra_args:
        print(f"Extra tau2 args: {' '.join(extra_args)}")
    print("=" * 44)

    command = [
        "tau2",
        "run",
        "--domain", args.domain,
        "--agent-llm", agent_llm,
        "--agent-llm-args", json.dumps({"temperature": args.temperature}),
        "--user-llm", user_llm,
        "--num-trials", str(args.num_trials),
        "--max-concurrency", str(args.max_concurrency),
    ]
    if args.num_tasks is not None:
        command += ["--num-tasks", str(args.num_tasks)]
    if args.save_to:
        command += ["--save-to", args.save_to]
    command += extra_args

    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"ERROR: tau2 run exited with code {result.returncode}")
        return result.returncode

    print("=== tau2 evaluation completed successfully ===")
    if args.output_dir:
        print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
