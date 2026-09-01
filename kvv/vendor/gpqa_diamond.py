import random

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
from inspect_ai.solver import TaskState, generate

from mmmu_pro_vision import parse_multi_choice_response

# Gated dataset: accept the terms on Hugging Face and set HF_TOKEN for the
# first download (the image build prefetch caches it after that).
DATASET_PATH = "Idavidrein/gpqa"
DATASET_NAME = "gpqa_diamond"

GPQA_PROMPT = (
    "Answer the following multiple-choice question. The last line of your response should be of "
    "the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
    "Think step by step before answering."
)

LETTERS = ["A", "B", "C", "D"]


def record_to_sample(record: dict) -> Sample:
    correct = record["Correct Answer"].strip()
    choices = [
        correct,
        record["Incorrect Answer 1"].strip(),
        record["Incorrect Answer 2"].strip(),
        record["Incorrect Answer 3"].strip(),
    ]

    # Shuffle deterministically per record so the correct answer's position
    # varies across questions but runs stay reproducible.
    record_id = str(record["Record ID"])
    random.Random(record_id).shuffle(choices)

    index2ans = dict(zip(LETTERS, choices))
    target = LETTERS[choices.index(correct)]

    options = "\n".join(f"{letter}. {ans}" for letter, ans in index2ans.items())
    prompt = f"{record['Question'].strip()}\n\n{options}\n\n{GPQA_PROMPT}"

    return Sample(
        id=record_id,
        input=prompt,
        target=target,
        metadata={
            "all_choices": list(LETTERS),
            "index2ans": index2ans,
        },
    )


@scorer(metrics=[accuracy(), stderr()])
def gpqa_diamond_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        target_answer = target.text.strip().upper()

        metadata = state.metadata or {}
        all_choices = metadata.get("all_choices", [])
        index2ans = metadata.get("index2ans", {})

        predicted = parse_multi_choice_response(completion, all_choices, index2ans)

        correct = predicted == target_answer
        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=predicted,
            explanation=f"Predicted={predicted}, Target={target_answer}",
        )

    return score


@task
def gpqa_diamond() -> Task:
    return Task(
        dataset=hf_dataset(
            path=DATASET_PATH,
            name=DATASET_NAME,
            split="train",
            sample_fields=record_to_sample,
        ),
        solver=[generate()],
        scorer=gpqa_diamond_scorer(),
    )
