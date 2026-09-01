import argparse, os

from inspect_ai import eval

from aime2025 import aime2025
from bfcl_multi_turn import MAXIMUM_STEP_LIMIT, bfcl_multi_turn
from gpqa_diamond import gpqa_diamond
from gsm8k import gsm8k
from mmmu_pro_vision import mmmu_pro_10c
from ocr_bench import ocrbench


import kimi_model  # noqa: F401 - registers kimi model API


BENCHMARKS = {
    "ocrbench": ocrbench,
    "mmmu": mmmu_pro_10c,
    "aime2025": aime2025,
    "bfcl": bfcl_multi_turn,
    "gsm8k": gsm8k,
    "gpqa": gpqa_diamond,
}

# Default configs per benchmark (max_connections, epochs)
BENCH_CONFIGS = {
    "ocrbench": {"max_connections": 50, "epochs": 1},
    "mmmu": {"max_connections": 50, "epochs": 1},
    "aime2025": {"max_connections": 50, "epochs": 32},
    "bfcl": {"max_connections": 32, "epochs": 1},
    "gsm8k": {"max_connections": 50, "epochs": 1},
    "gpqa": {"max_connections": 50, "epochs": 1},
}


def get_thinking_extra_body(
    thinking: bool,
    mode: str,
    thinking_effort: str | None = None,
) -> dict:
    """Build extra_body for thinking mode based on backend type.

    Args:
        thinking: Enable thinking mode
        mode: Backend type - "kimi", "opensource", or "none" (no thinking param)
        thinking_effort: Thinking effort to send when thinking is enabled
    """
    if mode == "none":
        # Non-hybrid model, no thinking param needed
        return {}
    elif mode == "opensource":
        # Open-source inference frameworks (vLLM, SGLang, KTransformers, etc.)
        if thinking:
            chat_template_kwargs = {"thinking": True}
            if thinking_effort:
                chat_template_kwargs["preserve_thinking"] = True
                chat_template_kwargs["thinking_effort"] = thinking_effort
            return {"chat_template_kwargs": chat_template_kwargs}
        else:
            return {"chat_template_kwargs": {"thinking": False}}
    else:  # kimi
        thinking_body = {"type": "enabled" if thinking else "disabled"}
        if thinking and thinking_effort:
            thinking_body["keep"] = "all"
            thinking_body["effort"] = thinking_effort
        return {"thinking": thinking_body}


def run_eval(
    bench_name: str,
    model: str,
    max_tokens: int,
    thinking: bool,
    think_mode: str,
    client_timeout: int,
    stream: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    thinking_effort: str | None = None,
    log_dir: str | None = None,
    log_format: str | None = None,
    log_buffer: int | None = None,
    display: str | None = None,
    limit: int | None = None,
    task_args: dict | None = None,
    **overrides,
):
    """Run a single benchmark evaluation."""
    task = BENCHMARKS[bench_name]
    config = BENCH_CONFIGS[bench_name]
    task_args = task_args or {}

    max_connections = overrides.get("max_connections", config["max_connections"])
    epochs = overrides.get("epochs", config["epochs"])

    extra_body = get_thinking_extra_body(thinking, think_mode, thinking_effort)

    print(f"\n{'='*60}")
    print(f"Running: {bench_name} | thinking={thinking} | mode={think_mode}")
    print(f"Model: {model}")
    print(f"max_tokens={max_tokens}, max_connections={max_connections}, epochs={epochs}")
    print(f"temperature={temperature}, top_p={top_p}")
    print(f"thinking_effort={thinking_effort}")
    print(f"stream={stream}, extra_body={extra_body}")
    print(f"task_args={task_args}")
    print(f"{'='*60}\n")

    print(f"{model=}")

    eval(
        [task],
        [model],
        task_args=task_args,
        max_tokens=max_tokens,
        max_connections=max_connections,
        epochs=epochs,
        extra_body=extra_body,
        continue_on_fail=True,
        fail_on_error=False,
        log_dir=log_dir,
        log_format=log_format,
        log_buffer=log_buffer,
        display=display,
        limit=limit,
        temperature=temperature,
        top_p=top_p,
        model_args={
            "stream": stream,
            "max_retries": 0,
            # inspect-ai>=0.3.258 sets the OpenAI client timeout itself from
            # client_timeout; passing a raw "timeout" model_arg now collides.
            "client_timeout": client_timeout,
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="Kimi Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "bench",
        nargs="?",
        choices=list(BENCHMARKS.keys()),
        default="ocrbench",
        help="Benchmark to run (default: ocrbench)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier (e.g., kimi/your-model-id)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        help="Max output tokens (see README for recommended values per benchmark)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode (requires --think-mode for hybrid models)",
    )
    parser.add_argument(
        "--think-mode",
        choices=["none", "kimi", "opensource"],
        default="none",
        help="Thinking param format: kimi (SaaS API) or opensource (vLLM/SGLang/KTransformers) (default: kimi)",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        help="Max concurrent connections",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of sampling epochs",
    )
    parser.add_argument(
        "--client-timeout",
        type=int,
        default=86400,
        help="HTTP request timeout in seconds (default: 86400)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming (keeps connection alive for long inference)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature (default: 1.0 for thinking, 0.6 for non-thinking)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p sampling (default: 0.95)",
    )
    parser.add_argument(
        "--thinking-effort",
        type=str,
        default=None,
        help=(
            "Kimi-style thinking effort, e.g. max. When set with "
            "--thinking --think-mode kimi, sends thinking.keep=all and "
            "thinking.effort=<value>."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N samples (smoke test; default: all)",
    )
    parser.add_argument(
        "--step-limit",
        type=int,
        default=None,
        help=(
            "bfcl only: model generations allowed per user turn before the "
            f"turn is force-quit (default: {MAXIMUM_STEP_LIMIT}, matching "
            "official BFCL's MAXIMUM_STEP_LIMIT). 0 restores inspect_evals' "
            "unbounded tool-calling loop."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for eval logs (default: INSPECT_LOG_DIR or ./logs)",
    )
    parser.add_argument(
        "--log-format",
        choices=["eval", "json"],
        default=os.environ.get("INSPECT_LOG_FORMAT", "eval"),
        # inspect_ai only honours INSPECT_LOG_FORMAT in its own CLI; the
        # Python eval() API hardcodes "eval", so read the env var ourselves.
        help="Log file format (default: $INSPECT_LOG_FORMAT or eval)",
    )
    parser.add_argument(
        "--log-buffer",
        type=int,
        default=1,
        help="Samples to buffer before flushing the log (default: 10; use 1 for live partial results)",
    )
    parser.add_argument(
        "--display",
        choices=["full", "conversation", "rich", "plain", "log", "none"],
        default=None,
        help="Progress display (default: full; use plain for non-TTY logs)",
    )
    args = parser.parse_args()

    overrides = {}
    if args.max_connections is not None:
        overrides["max_connections"] = args.max_connections
    if args.epochs is not None:
        overrides["epochs"] = args.epochs

    # --step-limit is a bfcl task arg; other tasks don't accept it.
    task_args = {}
    if args.step_limit is not None:
        if args.bench != "bfcl":
            parser.error("--step-limit only applies to the bfcl benchmark")
        task_args["step_limit"] = args.step_limit

    run_eval(
        args.bench,
        args.model,
        args.max_tokens,
        args.thinking,
        args.think_mode,
        args.client_timeout,
        args.stream,
        args.temperature,
        args.top_p,
        args.thinking_effort,
        log_dir=args.log_dir,
        log_buffer=args.log_buffer,
        log_format=args.log_format,
        display=args.display,
        limit=args.limit,
        task_args=task_args,
        **overrides,
    )


if __name__ == "__main__":
    main()
