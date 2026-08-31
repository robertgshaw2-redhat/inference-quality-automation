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
) -> Task:
    """BFCL multi-turn evaluation.

    Args:
        categories: inspect_evals/bfcl category names; a list or a
            comma-separated string. Defaults to the four BFCL v3
            multi-turn categories (200 conversations each).
    """
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(",") if c.strip()]
    return bfcl(categories=list(categories))
