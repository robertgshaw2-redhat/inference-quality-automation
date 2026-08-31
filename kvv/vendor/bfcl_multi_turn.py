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

Per-turn step limit
-------------------
``inspect_evals``' multi-turn solver drives each user turn with
``generate(state, tool_calls="loop")``, which only terminates when the
model finally answers without calling a tool. Official BFCL instead
force-quits a turn after ``MAXIMUM_STEP_LIMIT`` steps (gorilla@dac44e7,
``bfcl_eval/constants/default_prompts.py``, enforced in
``model_handler/base_handler.py``); the Inspect port dropped that cap, so
a model that loops on one tool runs until the eval itself is killed. We
restore the cap by handing the solver a ``generate`` that expands
``tool_calls="loop"`` into a bounded sequence of ``tool_calls="single"``
steps.
"""

from typing import Any, Literal

from inspect_ai import Task, task, task_with
from inspect_ai.log import transcript
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_evals.bfcl import bfcl
from inspect_evals.bfcl.bfcl import bfcl_solver

MULTI_TURN_CATEGORIES = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]

# Model steps (generations) allowed per user turn before the turn is cut
# short, matching official BFCL's MAXIMUM_STEP_LIMIT.
MAXIMUM_STEP_LIMIT = 20


def _called_tools(state: TaskState) -> bool:
    """Whether the most recent assistant message requested tool calls."""
    for message in reversed(state.messages):
        if isinstance(message, ChatMessageAssistant):
            return bool(message.tool_calls)
    return False


def _step_limited(generate: Generate, step_limit: int) -> Generate:
    """Wrap ``generate`` so a ``"loop"`` call runs at most ``step_limit`` steps.

    Each step is one model generation plus the execution of whatever tools it
    called — the same unit official BFCL counts. Turns that stop on their own
    (a text-only reply) behave exactly as before; only runaway turns differ.
    """

    async def step_limited_generate(
        state: TaskState,
        tool_calls: Literal["loop", "single", "none"] = "loop",
        **kwargs: Any,
    ) -> TaskState:
        if tool_calls != "loop":
            return await generate(state, tool_calls=tool_calls, **kwargs)

        for _ in range(step_limit):
            state = await generate(state, tool_calls="single", **kwargs)
            # A text-only response ends the turn, as does a sample-level limit.
            if state.completed or not _called_tools(state):
                return state

        transcript().info(
            f"Model has been forced to quit after {step_limit} steps.",
            source="bfcl_step_limit",
        )
        return state

    return step_limited_generate


@solver
def step_limited_bfcl_solver(step_limit: int = MAXIMUM_STEP_LIMIT) -> Solver:
    """``inspect_evals``' BFCL solver with a per-turn step cap.

    Args:
        step_limit: Model generations allowed per user turn. 0 (or negative)
            restores the unbounded upstream behaviour.
    """
    inner = bfcl_solver()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if step_limit <= 0:
            return await inner(state, generate)
        return await inner(state, _step_limited(generate, step_limit))

    return solve


@task
def bfcl_multi_turn(
    categories: str | list[str] = MULTI_TURN_CATEGORIES,
    step_limit: int = MAXIMUM_STEP_LIMIT,
) -> Task:
    """BFCL multi-turn evaluation.

    Args:
        categories: inspect_evals/bfcl category names; a list or a
            comma-separated string. Defaults to the four BFCL v3
            multi-turn categories (200 conversations each).
        step_limit: Model generations allowed per user turn before the turn
            is force-quit, matching official BFCL's MAXIMUM_STEP_LIMIT of 20.
            Pass 0 for upstream's unbounded loop.
    """
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(",") if c.strip()]
    return task_with(
        bfcl(categories=list(categories)),
        solver=step_limited_bfcl_solver(step_limit),
    )
