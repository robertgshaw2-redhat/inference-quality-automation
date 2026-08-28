"""Schema-fuzz suite: force a tool call per JSON Schema, validate arguments.

Each selected walle-valid schema is sent as a tool's ``parameters`` with
``tool_choice="required"``; the returned ``function.arguments`` must be
valid JSON that validates against the schema — in both non-streaming and
streaming modes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openai
import pytest

from tcv.openai_compat import request_completion
from tcv.schema_cases import (
    REQUEST_MODES,
    ValidatorCase,
    build_request,
    extract_case_arguments,
    load_cases,
    select_cases,
    validate_arguments,
)


def _selected(config: pytest.Config) -> list[tuple[ValidatorCase, Any, str]]:
    cases = load_cases(Path(config.getoption("schema_case_dir")))
    return select_cases(cases, max_cases=config.getoption("max_cases"))


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "schema_case_info" not in metafunc.fixturenames:
        return
    params = [
        pytest.param(
            (case, schema, reason, mode),
            id=f"{case.suite}:{case.line}:{mode}",
        )
        for case, schema, reason in _selected(metafunc.config)
        for mode in REQUEST_MODES
    ]
    metafunc.parametrize("schema_case_info", params)


def test_tool_call_arguments_match_schema(
    client: openai.Client,
    model: str,
    extra_body: dict,
    default_max_tokens: int,
    schema_case_info: tuple[ValidatorCase, Any, str, str],
) -> None:
    case, schema, selection_reason, mode = schema_case_info
    result = request_completion(
        client,
        model,
        build_request(schema),
        stream=mode == "stream",
        max_tokens=default_max_tokens,
        extra_body=extra_body,
    )
    arguments, message = extract_case_arguments(result)
    valid, validation_message = validate_arguments(schema, arguments)
    assert valid, (
        f"{case.suite}:{case.line} [{mode}] ({selection_reason}) "
        f"{validation_message}; extraction: {message}"
    )
