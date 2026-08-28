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
        "--num-threads",
        type=int,
        default=32,
        help="Threads for BFCL generation (default: 8)",
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
        OpenAIResponsesHandler
        if api_type == "responses"
        else OpenAICompletionsHandler
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

    try:
        if not args.skip_generate:
            print(f"=== BFCL generate: model={model} ===")
            gen_kwargs = default_kwargs(generate)
            gen_kwargs["model"] = [model]
            gen_kwargs["test_category"] = args.test_category
            gen_kwargs["skip_server_setup"] = True
            gen_kwargs["num_threads"] = args.num_threads
            gen_kwargs["temperature"] = args.temperature
            generate(**gen_kwargs)

        if not args.skip_evaluate:
            print(f"=== BFCL evaluate: model={model} ===")
            eval_kwargs = default_kwargs(evaluate)
            eval_kwargs["model"] = [model]
            eval_kwargs["test_category"] = args.test_category
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
