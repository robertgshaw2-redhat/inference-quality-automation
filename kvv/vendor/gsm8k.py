from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from aime2025 import verify_answer

DATASET_PATH = "openai/gsm8k"

# The dataset's answer field is chain-of-thought reasoning, then this
# delimiter, then the final numeric answer (which may contain thousands
# separators, e.g. "#### 1,000").
ANSWER_DELIM = "####"


def record_to_sample(record: dict) -> Sample:
    target = record["answer"].split(ANSWER_DELIM)[-1]
    return Sample(
        input=record["question"],
        target=target.strip().replace(",", ""),
    )


@solver
def gsm8k_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.user_prompt:
            state.user_prompt.text = f"{state.user_prompt.text}\n\n Please reason step by step, and put your final answer within \\boxed{{}}."
        return await generate(state)

    return solve


@scorer(metrics=[accuracy(), stderr()])
def gsm8k_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        target_val = target.text.strip()
        extracted_answer, correct = verify_answer(state.output.completion, target_val)
        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=str(extracted_answer),
            explanation=f"Extracted={extracted_answer}, Target={target_val}",
        )

    return score


@task
def gsm8k() -> Task:
    return Task(
        dataset=hf_dataset(
            path=DATASET_PATH,
            name="main",
            split="test",
            sample_fields=record_to_sample,
        ),
        solver=gsm8k_solver(),
        scorer=gsm8k_scorer(),
    )
