"""Data-driven tool-call behavior tests.

Every case in cases/*.jsonl runs in each of its modes (non-stream and
stream by default). See tcv/cases.py for the case format.
"""

from __future__ import annotations

from pathlib import Path

import openai
import pytest

from tcv.cases import BehaviorCase, evaluate_case, load_cases, select_cases
from tcv.openai_compat import request_completion


def _selected_cases(config: pytest.Config) -> list[BehaviorCase]:
    cases = load_cases(Path(config.getoption("cases_dir")))
    tags = {t.strip() for t in config.getoption("tags").split(",") if t.strip()}
    return select_cases(
        cases,
        tags=tags or None,
        id_filter=config.getoption("case_filter") or None,
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "case_run" not in metafunc.fixturenames:
        return
    params = [
        pytest.param((case, mode), id=f"{case.id}:{mode}")
        for case in _selected_cases(metafunc.config)
        for mode in case.modes
    ]
    metafunc.parametrize("case_run", params)


def test_behavior_case(
    client: openai.Client,
    model: str,
    extra_body: dict,
    default_max_tokens: int,
    case_run: tuple[BehaviorCase, str],
) -> None:
    case, mode = case_run
    result = request_completion(
        client,
        model,
        case.request,
        stream=mode == "stream",
        max_tokens=default_max_tokens,
        extra_body=extra_body,
    )
    outcome = evaluate_case(case, result)
    assert outcome.passed, (
        f"{case.id} [{mode}] ({case.source}) {case.description}\n"
        + "\n".join(f"  - {f}" for f in outcome.failures)
        + f"\n  response: {result.describe()}"
    )
