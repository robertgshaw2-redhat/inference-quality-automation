#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run Terminal-Bench 2.1 against an *already running* OpenAI-compatible
server, using Harbor (https://harborframework.com) as the eval harness.

Terminal-Bench 2.1 (https://www.tbench.ai) is 89 hard, realistic terminal
tasks (build this project, fix this async bug, assemble this genome, ...).
For each task Harbor builds the task's Docker environment, drops the agent
into a tmux session inside it, and afterwards runs the task's verifier tests;
the reward is 1 only if the tests pass. The default agent is Terminus 2, the
benchmark's reference agent: it runs in *this* process and talks to the
container over tmux, so only this harness (not the task containers) needs to
reach the model server.

The task containers run on the Docker daemon at /var/run/docker.sock. When
this script itself runs inside a container against the host's socket, the
jobs dir must be visible at the *same absolute path* on the host and in the
container, because Harbor bind-mounts trial dirs into the task containers
(the Justfile targets set this up).

Examples:
    # Server on localhost:8000, model auto-detected from /v1/models,
    # all 89 tasks
    python run_terminalbench_eval.py

    # Smoke test: 3 tasks, 2 at a time
    python run_terminalbench_eval.py --n-tasks 3 --n-concurrent 2

    # A single task, plus pass@k over 4 attempts
    python run_terminalbench_eval.py --task hello-world --n-attempts 4

Any unrecognized arguments are forwarded verbatim to `harbor run`
(see `harbor run --help`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_DATASET_PATH = os.environ.get(
    "TB_DATASET_PATH", "/opt/terminal-bench/terminal-bench-2-1"
)


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
        "--agent",
        "-a",
        default="terminus-2",
        help="Harbor agent to run (default: terminus-2, the Terminal-Bench "
        "reference agent). For non-terminus agents the model endpoint kwargs "
        "are not injected; pass the agent's own --ak/--ae flags instead.",
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
        help="Local dataset directory of Terminal-Bench tasks "
        f"(default: {DEFAULT_DATASET_PATH}, baked into the image)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default=None,
        help="Registry dataset name@version (e.g. "
        "terminal-bench/terminal-bench-2-1); overrides --dataset-path and "
        "downloads from the Harbor registry instead",
    )
    parser.add_argument(
        "--task",
        "-t",
        action="append",
        default=None,
        help="Run only tasks matching this name (glob ok; repeatable)",
    )
    parser.add_argument(
        "--exclude-task",
        "-x",
        action="append",
        default=None,
        help="Skip tasks matching this name (glob ok; repeatable)",
    )
    parser.add_argument(
        "--n-tasks",
        "-l",
        type=int,
        default=None,
        help="Run only the first N tasks after filtering (default: all 89)",
    )
    parser.add_argument(
        "--n-attempts",
        "-k",
        type=int,
        default=1,
        help="Attempts per task; >1 enables pass@k metrics (default: 1)",
    )
    parser.add_argument(
        "--n-concurrent",
        "-n",
        type=int,
        default=4,
        help="Concurrent trials, each with its own task container "
        "(default: 4)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Cap on agent turns per task (default: unlimited; tasks are "
        "bounded by their own time limits)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for the agent (default: server default)",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="Model context length, used by Terminus for context management. "
        "Default: max_model_len reported by the server's /v1/models (vLLM), "
        "else 131072.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Max completion tokens registered for the model "
        "(default: --max-input-tokens)",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Run name; results land in <jobs-dir>/<job-name>/ "
        "(default: tb21-<timestamp>)",
    )
    parser.add_argument(
        "--jobs-dir",
        default="/results",
        help="Directory for Harbor job results (default: /results). With a "
        "mounted docker socket this must be the same absolute path on the "
        "host and in this container.",
    )
    return parser.parse_known_args()


def detect_model(base_url: str, api_key: str) -> tuple[str, int | None]:
    """Ask the running server which model it serves and its context length.

    vLLM reports max_model_len per model in /v1/models; other servers may
    not, in which case the context length falls back to a flag/default.
    """
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
    max_model_len = models[0].get("max_model_len")
    return models[0]["id"], max_model_len if isinstance(max_model_len, int) else None


def check_docker() -> None:
    """Fail fast with a useful message when the docker socket is missing."""
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        sys.exit(f"ERROR: docker CLI not runnable ({exc}).")
    if result.returncode != 0:
        sys.exit(
            "ERROR: cannot reach a Docker daemon "
            f"({result.stderr.strip() or result.stdout.strip()}).\n"
            "Terminal-Bench runs each task in its own container; mount the "
            "host's /var/run/docker.sock into this container (see the "
            "`just terminalbench` target)."
        )


def summarize(job_dir: Path) -> None:
    """Print the headline numbers from Harbor's result.json."""
    result_path = job_dir / "result.json"
    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not read {result_path} ({exc})")
        return

    stats = result.get("stats") or {}
    print(f"Trials completed: {stats.get('n_completed_trials')}")
    if stats.get("n_errored_trials"):
        print(f"Trials errored:   {stats.get('n_errored_trials')}")
    for name, ev in (stats.get("evals") or {}).items():
        print(f"[{name}]")
        for metric in ev.get("metrics") or []:
            print(f"  {json.dumps(metric)}")
        for k, value in (ev.get("pass_at_k") or {}).items():
            print(f"  pass@{k}: {value}")
    print(f"Results: {job_dir}")


def main() -> int:
    args, extra_args = parse_args()
    args.base_url = args.base_url.rstrip("/")

    check_docker()

    model, max_model_len = detect_model(args.base_url, args.api_key)
    # LiteLLM's openai/ prefix routes to any OpenAI-compatible endpoint; the
    # endpoint itself is passed to Terminus via its api_base kwarg.
    agent_model = model if model.startswith("openai/") else f"openai/{model}"

    max_input_tokens = args.max_input_tokens or max_model_len or 131072
    max_output_tokens = args.max_output_tokens or max_input_tokens
    # Registering the model with LiteLLM (via Terminus's model_info kwarg)
    # gives Terminus the real context limit for its summarization logic and
    # a zero cost entry so cost accounting doesn't error on a local model.
    model_info = {
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "openai",
        "mode": "chat",
    }

    # For the openai/ LiteLLM provider and for CLI agents that read the
    # standard env vars.
    os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["OPENAI_BASE_URL"] = args.base_url
    os.environ["OPENAI_API_BASE"] = args.base_url

    job_name = args.job_name or time.strftime("tb21-%Y%m%d-%H%M%S")
    jobs_dir = Path(args.jobs_dir)

    command = [
        "harbor",
        "run",
        "--agent", args.agent,
        "--model", agent_model,
        "--n-concurrent", str(args.n_concurrent),
        "--n-attempts", str(args.n_attempts),
        "--jobs-dir", str(jobs_dir),
        "--job-name", job_name,
        "--yes",
    ]
    if args.dataset:
        command += ["--dataset", args.dataset]
    else:
        command += ["--path", args.dataset_path]
    if args.agent.startswith("terminus"):
        command += ["--ak", f"api_base={args.base_url}"]
        command += ["--ak", f"model_info={json.dumps(model_info)}"]
        if args.max_turns is not None:
            command += ["--ak", f"max_turns={args.max_turns}"]
        if args.temperature is not None:
            command += ["--ak", f"temperature={args.temperature}"]
    for task in args.task or []:
        command += ["--include-task-name", task]
    for task in args.exclude_task or []:
        command += ["--exclude-task-name", task]
    if args.n_tasks is not None:
        command += ["--n-tasks", str(args.n_tasks)]
    command += extra_args

    print("=" * 44)
    print("Terminal-Bench 2.1 Evaluation (Harbor)")
    print("=" * 44)
    print(f"Base URL:        {args.base_url}")
    print(f"Agent:           {args.agent}")
    print(f"Agent model:     {agent_model}")
    print(f"Context length:  {max_input_tokens}")
    print(f"Dataset:         {args.dataset or args.dataset_path}")
    print(f"Tasks:           {args.n_tasks if args.n_tasks else 'all'}")
    print(f"Attempts/task:   {args.n_attempts}")
    print(f"Concurrency:     {args.n_concurrent}")
    print(f"Job:             {jobs_dir / job_name}")
    if extra_args:
        print(f"Extra args:      {' '.join(extra_args)}")
    print("=" * 44)

    returncode = subprocess.call(command)
    if returncode != 0:
        print(f"ERROR: harbor run exited with code {returncode}")
        summarize(jobs_dir / job_name)
        return returncode

    print("=== Terminal-Bench evaluation completed successfully ===")
    summarize(jobs_dir / job_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
