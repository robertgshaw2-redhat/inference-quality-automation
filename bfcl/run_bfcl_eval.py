#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run BFCL (Berkeley Function Call Leaderboard) tool-calling correctness
evaluation against an *already running* OpenAI-compatible server.

Unlike .buildkite/scripts/tool_call/run-bfcl-eval.sh, this script never starts
or stops a server; point it at one you launched yourself.

Requires: uv pip install "bfcl-eval>=2025.10.20.1,<2026"

Examples:
    # Server on localhost:8000, model auto-detected from /v1/models
    python run_bfcl_eval.py

    # Explicit model / categories / remote server
    python run_bfcl_eval.py \
        --base-url http://my-host:8000/v1 \
        --model openai/gpt-oss-120b \
        --test-category live_simple multiple parallel_multiple \
        --output-dir ./bfcl-results

    # Responses API instead of chat completions
    python run_bfcl_eval.py --api-type responses --output-dir ./bfcl-responses

    # Smoke test: only the first 5 entries of each test category
    python run_bfcl_eval.py --num-prompts 5
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import urllib.error
import urllib.request


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
        "--api-type",
        choices=["chat_completions", "responses", "messages"],
        default="chat_completions",
        help="Which API surface to exercise (default: chat_completions)",
    )
    parser.add_argument(
        "--test-category",
        nargs="+",
        default=["multi_turn"],
        help="BFCL test categories (default: multi_turn). Accepts multiple "
        "space-separated values or a single comma-separated string.",
    )
    parser.add_argument(
        "--num-prompts",
        "-n",
        type=int,
        default=None,
        help="Run only the first N test entries of each category instead of "
        "the full set. Implies --partial-eval.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=32,
        help="Threads for BFCL generation (default: 32)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for BFCL result/ and score/ output (default: cwd)",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Only score existing results in --output-dir",
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Only generate responses, do not score them",
    )
    parser.add_argument(
        "--partial-eval",
        action="store_true",
        help="Score whatever entries exist in the result files instead of "
        "failing on missing ids. Set automatically by --num-prompts.",
    )
    parser.add_argument(
        "--no-underscore-to-dot",
        action="store_true",
        help="Disable BFCL's underscore-to-dot function name conversion",
    )
    args = parser.parse_args()

    # Allow --test-category "a, b" in addition to --test-category a b
    categories: list[str] = []
    for chunk in args.test_category:
        categories.extend(c.strip() for c in chunk.split(",") if c.strip())
    args.test_category = categories
    args.base_url = args.base_url.rstrip("/")
    if args.num_prompts is not None and args.num_prompts < 1:
        parser.error("--num-prompts must be >= 1")
    if args.num_prompts is not None:
        args.partial_eval = True
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


def register_model(model: str, api_type: str, underscore_to_dot: bool) -> None:
    """Teach BFCL to route this model to the OpenAI-compatible handler."""
    import bfcl_eval.constants.model_config as bfcl_model_config
    from bfcl_eval.constants.model_config import ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )
    from bfcl_eval.model_handler.api_inference.openai_response import (
        OpenAIResponsesHandler,
    )

    handler = (
        OpenAIResponsesHandler if api_type == "responses" else OpenAICompletionsHandler
    )

    bfcl_model_config.MODEL_CONFIG_MAPPING[model] = ModelConfig(
        model_name=model,
        display_name=f"{model} (FC) (vLLM)",
        url=f"https://huggingface.co/{model}",
        org="",
        license="apache-2.0",
        model_handler=handler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=underscore_to_dot,
    )


def write_test_id_subset(
    categories: list[str], num_prompts: int, output_dir: str
) -> str:
    """Write BFCL's test_case_ids_to_generate.json for the first N entries.

    BFCL only supports subsetting through an id file at the project root,
    selected with generate's ``--run-ids``.

    Args:
        categories: Test categories or category groups from --test-category.
        num_prompts: Number of entries to keep per expanded category.
        output_dir: BFCL project root; the id file is written here.

    Returns:
        Path of the id file that was written.
    """
    from bfcl_eval.utils import (
        is_memory_prereq,
        load_dataset_entry,
        parse_test_category_argument,
    )

    subset: dict[str, list[str]] = {}
    for category in parse_test_category_argument(categories):
        entries = load_dataset_entry(category)
        by_id = {entry["id"]: entry for entry in entries}
        selected = [
            entry["id"] for entry in entries if not is_memory_prereq(entry["id"])
        ][:num_prompts]

        # Memory entries cannot run without their prerequisite conversations.
        keep = set(selected)
        pending = list(selected)
        while pending:
            entry = by_id.get(pending.pop(), {})
            for dep in entry.get("depends_on", []):
                if dep not in keep:
                    keep.add(dep)
                    pending.append(dep)

        subset[category] = [entry["id"] for entry in entries if entry["id"] in keep]
        print(f"  {category}: {len(subset[category])} of {len(entries)} entries")

    path = os.path.join(output_dir, "test_case_ids_to_generate.json")
    with open(path, "w") as id_file:
        json.dump(subset, id_file, indent=2)
    return path


def default_kwargs(function) -> dict:
    """Extract default values from a Typer command's signature."""
    import typer

    kwargs = {}
    for name, param in inspect.signature(function).parameters.items():
        if param.default is inspect.Parameter.empty:
            continue
        default = param.default
        if isinstance(default, typer.models.OptionInfo):
            default = default.default
        kwargs[name] = default
    return kwargs


def main() -> int:
    args = parse_args()

    model = args.model or detect_model(args.base_url, args.api_key)
    output_dir = os.path.abspath(args.output_dir or os.getcwd())
    os.makedirs(output_dir, exist_ok=True)

    # These must be set before importing bfcl_eval: the handlers read them at
    # import time to build their OpenAI clients, and BFCL_PROJECT_ROOT decides
    # where result/ and score/ are written.
    os.environ["OPENAI_BASE_URL"] = args.base_url
    os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["BFCL_PROJECT_ROOT"] = output_dir

    print("=" * 44)
    print("BFCL Tool Call Correctness Evaluation")
    print("=" * 44)
    print(f"Base URL:       {args.base_url}")
    print(f"Model:          {model}")
    print(f"API type:       {args.api_type}")
    print(f"Output dir:     {output_dir}")
    print(f"Test category:  {', '.join(args.test_category)}")
    if args.num_prompts is not None:
        print(f"Num prompts:    {args.num_prompts} per category")
    print(f"Num threads:    {args.num_threads}")
    print(f"Temperature:    {args.temperature}")
    print("=" * 44)

    try:
        register_model(model, args.api_type, not args.no_underscore_to_dot)
    except ImportError as exc:
        sys.exit(
            f"ERROR: {exc}\n"
            'Install it with: uv pip install "bfcl-eval>=2025.10.20.1,<2026"'
        )

    from bfcl_eval.__main__ import evaluate, generate

    if args.num_prompts is not None:
        print(f"=== BFCL subset: first {args.num_prompts} entries per category ===")
        id_file = write_test_id_subset(args.test_category, args.num_prompts, output_dir)
        print(f"Test id file:   {id_file}")

    try:
        if not args.skip_generate:
            print(f"=== BFCL generate: model={model} ===")
            gen_kwargs = default_kwargs(generate)
            gen_kwargs["model"] = [model]
            gen_kwargs["test_category"] = args.test_category
            gen_kwargs["skip_server_setup"] = True
            gen_kwargs["num_threads"] = args.num_threads
            gen_kwargs["temperature"] = args.temperature
            if args.num_prompts is not None:
                # BFCL reads the ids from test_case_ids_to_generate.json and
                # ignores --test-category when this is set.
                gen_kwargs["run_ids"] = True
            generate(**gen_kwargs)

        if not args.skip_evaluate:
            print(f"=== BFCL evaluate: model={model} ===")
            eval_kwargs = default_kwargs(evaluate)
            eval_kwargs["model"] = [model]
            eval_kwargs["test_category"] = args.test_category
            if args.partial_eval:
                eval_kwargs["partial_eval"] = True
            evaluate(**eval_kwargs)
    finally:
        # filelock artifacts from BFCL's thread-safe writes
        for lock_dir in (".file_locks", os.path.join(output_dir, ".file_locks")):
            shutil.rmtree(lock_dir, ignore_errors=True)

    print("=== BFCL evaluation completed successfully ===")
    print(f"Results: {os.path.join(output_dir, 'result')}")
    print(f"Scores:  {os.path.join(output_dir, 'score')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
