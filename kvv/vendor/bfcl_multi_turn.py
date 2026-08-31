"""BFCL v3 multi-turn tool-calling benchmark.

Thin wrapper around the ``inspect_evals/bfcl`` task (UKGovernmentBEIS/
inspect_evals), which implements BFCL's stateful multi-turn categories
(GorillaFileSystem, TradingBot, VehicleControl, ... backends with
state-based + response-based per-turn checking) on Inspect AI.

The default categories are BFCL's leaderboard ``multi_turn`` group; the
scorer reports overall accuracy plus grouped per-category accuracy
(``multi_turn_base_acc`` etc.). ``multi_turn_composite`` exists in the
dataset but is not part of the leaderboard group, so it is excluded by
default; pass it explicitly if you want it.
"""

from inspect_ai import Task, task
from inspect_evals.bfcl import bfcl

MULTI_TURN_CATEGORIES = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]


@task
def bfcl_multi_turn(
    categories: str | list[str] = MULTI_TURN_CATEGORIES,
    message_limit: int | None = 300,
    time_limit: int | None = 3600,
) -> Task:
    """BFCL multi-turn evaluation.

    Args:
        categories: inspect_evals/bfcl category names; a list or a
            comma-separated string. Defaults to the four BFCL v3
            multi-turn categories (200 conversations each).
        message_limit: Per-sample cap on conversation length. The
            inspect_evals solver runs generate(tool_calls="loop") with no
            step cap (upstream BFCL stops after 20 steps per turn), so a
            model stuck re-issuing tool calls otherwise loops with an
            ever-growing context. A limited sample is scored on the turns
            it completed, matching upstream's forced termination. 300
            comfortably covers legitimate trajectories (median ~4 user
            turns, a handful of tool calls each).
        time_limit: Per-sample wall-clock cap in seconds, bounding hangs
            (the client's streaming read timeout is unlimited).
    """
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(",") if c.strip()]
    bfcl_task = bfcl(categories=list(categories))
    bfcl_task.message_limit = message_limit
    bfcl_task.time_limit = time_limit
    return bfcl_task
