#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the GSM8K grade-school math benchmark against an *already running*
OpenAI-compatible server, using Inspect AI (https://inspect.aisi.org.uk) as
the eval harness.

GSM8K (https://huggingface.co/datasets/openai/gsm8k) is 1319 grade-school
math word problems that take 2-8 arithmetic steps to solve. The model is
prompted with chain-of-thought few-shot examples (8-shot by default, the
canonical setup) and must end its response with "ANSWER: $ANSWER"; a sample
scores correct when that final answer numerically matches the reference.

Inspect AI drives the requests, retries, and scoring, and writes a full
transcript log per run; view logs afterwards with `inspect view`.

Examples:
    # Server on localhost:8000, model auto-detected from /v1/models,
    # full test split (1319 problems)
    python run_gsm8k_eval.py

    # Smoke test: 10 problems
    python run_gsm8k_eval.py --num-problems 10

    # Reliability signal: accuracy/stderr over 4 epochs, zero-shot
    python run_gsm8k_eval.py --epochs 4 --fewshot 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DATASET = "openai/gsm8k"
ANSWER_DELIM = "####"

# Prompt used by the reference GSM8K task in UK AISI's inspect_evals.
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your \
response should be of the form "ANSWER: $ANSWER" (without quotes) where \
$ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form \
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the \
problem, and you do not need to use a \\boxed command.
""".strip()

FEWSHOT_TEMPLATE = """
Here are some examples of how to solve similar problems:

{examples}
""".strip()


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
        "--num-problems",
        "-n",
        type=int,
        default=None,
        help="Run only the first N problems of the test split (default: all 1319)",
    )
    parser.add_argument(
        "--fewshot",
        type=int,
        default=8,
        help="Number of chain-of-thought few-shot examples drawn from the "
        "train split; 0 for zero-shot (default: 8)",
    )
    parser.add_argument(
        "--fewshot-seed",
        type=int,
        default=42,
        help="Seed for sampling the few-shot examples (default: 42)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Passes over the dataset; >1 averages accuracy across epochs "
        "(default: 1)",
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
        default=2048,
        help="Max completion tokens per problem (default: 2048)",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=100,
        help="Concurrent requests to the server (default: 32)",
    )
    parser.add_argument(
        "--log-dir",
        default=os.environ.get("INSPECT_LOG_DIR", "./logs"),
        help="Directory for Inspect AI eval logs (default: $INSPECT_LOG_DIR "
        "or ./logs)",
    )
    return parser.parse_args()


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


def record_to_sample(record):
    """Map a GSM8K record onto an Inspect Sample.

    The dataset's answer field is chain-of-thought reasoning, then '####',
    then the final numeric answer; only the final answer is the scoring
    target, the reasoning is kept for few-shot examples.
    """
    from inspect_ai.dataset import Sample

    *reasoning, target = record["answer"].split(ANSWER_DELIM)
    return Sample(
        input=record["question"],
        target=target.strip(),
        metadata={"reasoning": ANSWER_DELIM.join(reasoning).strip()},
    )


def sample_to_fewshot(sample) -> str:
    return (
        f"{sample.input}\n\nReasoning:\n{sample.metadata['reasoning']}"
        f"\n\nANSWER: {sample.target}"
    )


def gsm8k_task(num_problems: int | None, fewshot: int, fewshot_seed: int):
    from inspect_ai import Task
    from inspect_ai.dataset import hf_dataset
    from inspect_ai.scorer import match
    from inspect_ai.solver import generate, prompt_template, system_message

    solver = [prompt_template(MATH_PROMPT_TEMPLATE), generate()]
    if fewshot:
        examples = hf_dataset(
            DATASET,
            name="main",
            split="train",
            sample_fields=record_to_sample,
            shuffle=True,
            seed=fewshot_seed,
            limit=fewshot,
        )
        solver.insert(
            0,
            system_message(
                FEWSHOT_TEMPLATE.format(
                    examples="\n\n".join(sample_to_fewshot(s) for s in examples)
                )
            ),
        )

    return Task(
        name="gsm8k",
        dataset=hf_dataset(
            DATASET,
            name="main",
            split="test",
            sample_fields=record_to_sample,
            limit=num_problems,
        ),
        solver=solver,
        scorer=match(numeric=True),
    )


def main() -> int:
    args = parse_args()
    args.base_url = args.base_url.rstrip("/")

    model = args.model or detect_model(args.base_url, args.api_key)
    # openai-api/local/... is Inspect's generic OpenAI-compatible provider;
    # it reads the endpoint from the LOCAL_* variables set below.
    inspect_model = f"openai-api/local/{model}"
    os.environ["LOCAL_BASE_URL"] = args.base_url
    os.environ["LOCAL_API_KEY"] = args.api_key

    print("=" * 44)
    print("GSM8K Evaluation (Inspect AI)")
    print("=" * 44)
    print(f"Base URL:        {args.base_url}")
    print(f"Model:           {model}")
    print(f"Num problems:    {args.num_problems if args.num_problems else 'all'}")
    print(f"Fewshot:         {args.fewshot}")
    print(f"Epochs:          {args.epochs}")
    print(f"Temperature:     {args.temperature}")
    print(f"Max tokens:      {args.max_tokens}")
    print(f"Max connections: {args.max_connections}")
    print(f"Log dir:         {args.log_dir}")
    print("=" * 44)

    from inspect_ai import eval as inspect_eval

    logs = inspect_eval(
        gsm8k_task(args.num_problems, args.fewshot, args.fewshot_seed),
        model=inspect_model,
        epochs=args.epochs,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_connections=args.max_connections,
        log_dir=args.log_dir,
        display="plain",
    )

    log = logs[0]
    if log.status != "success" or log.results is None:
        print(f"ERROR: eval finished with status '{log.status}'")
        if log.error is not None:
            print(log.error.message)
        return 1

    print("=== GSM8K evaluation completed successfully ===")
    for score in log.results.scores:
        for name, metric in score.metrics.items():
            print(f"{name}: {metric.value:.4f}")
    print(f"Log: {log.location}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
