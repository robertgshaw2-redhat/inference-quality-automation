"""Data-driven behavior cases.

Cases live in ``cases/*.jsonl`` — one JSON object per line. Growing the
baseline over time means appending a line (or adding a file); no code
changes are needed. Each case is:

    {
      "id": "unique_case_id",
      "description": "what this case checks",
      "tags": ["basic"],                       # optional, for filtering
      "modes": ["non-stream", "stream"],       # optional, default both
      "request": {
        "messages": [...],                     # chat messages, verbatim
        "tools": [...],                        # OpenAI tool definitions
        "tool_choice": "auto",                 # optional
        "parallel_tool_calls": true,           # optional
        "max_tokens": 512                      # optional per-case override
      },
      "expect": { ... }                        # see below
    }

Expectations (``expect``):

    "tool_calls": {
      "min": 1,                # minimum number of calls (default 1)
      "max": 1,                # maximum number of calls (optional)
      "names": ["a", "b"],    # every call's name must be in this set
      "require_names": ["a"], # each listed name must appear at least once
      "args_match_schema": true,  # validate args against the tool's
                                  # declared parameters schema (default true)
      "checks": [              # assertions on parsed argument values
        {"path": "city", "contains_any": ["Tokyo"], "call": "any"}
      ]
    }

    "no_tool_calls": true      # instead of "tool_calls"

    "content": {               # assertions on message content
      "nonempty": true,
      "contains": ["x"],       # all must appear
      "contains_any": ["x", "y"]
    }

    "finish_reason": "stop"    # optional; "any" disables the default
                               # (default: "tool_calls" when tool calls are
                               # expected, "stop" when they are forbidden)

Argument checks: ``path`` is a dot path into the parsed arguments object
(numeric segments index arrays, e.g. "items.0.sku"). ``call`` selects which
tool call the check applies to: an integer index, or "any" (default) —
satisfied if any call passes. Exactly one op per check:

    "equals": value          strict equality (5 != 5.0 is not enforced;
                             bools are type-checked strictly)
    "one_of": [v1, v2]       equality with any listed value
    "contains": "sub"        substring of a string value
    "contains_any": [...]    any substring matches
    "json_type": "integer"   one of string/number/integer/boolean/object/
                             array/null
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from tcv.openai_compat import CompletionResult, ToolCall, truncate


VALID_MODES = ("non-stream", "stream")


@dataclass(frozen=True)
class BehaviorCase:
    id: str
    description: str
    request: dict[str, Any]
    expect: dict[str, Any]
    modes: tuple[str, ...]
    tags: tuple[str, ...]
    source: str  # "<file>:<line>"


def load_cases(cases_dir: Path) -> list[BehaviorCase]:
    if not cases_dir.is_dir():
        raise FileNotFoundError(f"cases directory not found: {cases_dir}")

    cases: list[BehaviorCase] = []
    seen_ids: dict[str, str] = {}
    for path in sorted(cases_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line_number, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                source = f"{path.name}:{line_number}"
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}: invalid JSON: {exc.msg}") from None
                case = _parse_case(data, source)
                if case.id in seen_ids:
                    raise ValueError(
                        f"{source}: duplicate case id {case.id!r} "
                        f"(first seen at {seen_ids[case.id]})"
                    )
                seen_ids[case.id] = source
                cases.append(case)
    return cases


def _parse_case(data: dict[str, Any], source: str) -> BehaviorCase:
    for key in ("id", "request", "expect"):
        if key not in data:
            raise ValueError(f"{source}: case is missing required key {key!r}")
    request = data["request"]
    if "messages" not in request:
        raise ValueError(f"{source}: request is missing 'messages'")
    modes = tuple(data.get("modes", VALID_MODES))
    for mode in modes:
        if mode not in VALID_MODES:
            raise ValueError(f"{source}: invalid mode {mode!r}")
    expect = data["expect"]
    if ("tool_calls" in expect) == bool(expect.get("no_tool_calls")):
        raise ValueError(
            f"{source}: expect must set exactly one of 'tool_calls' / "
            "'no_tool_calls'"
        )
    return BehaviorCase(
        id=str(data["id"]),
        description=str(data.get("description", "")),
        request=request,
        expect=expect,
        modes=modes,
        tags=tuple(data.get("tags", [])),
        source=source,
    )


def select_cases(
    cases: list[BehaviorCase],
    *,
    tags: set[str] | None = None,
    id_filter: str | None = None,
) -> list[BehaviorCase]:
    selected = cases
    if tags:
        selected = [c for c in selected if tags & set(c.tags)]
    if id_filter:
        selected = [c for c in selected if id_filter in c.id]
    return selected


# ---------------------------------------------------------------------------
# Expectation evaluation
# ---------------------------------------------------------------------------


@dataclass
class CheckOutcome:
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def evaluate_case(case: BehaviorCase, result: CompletionResult) -> CheckOutcome:
    outcome = CheckOutcome()
    expect = case.expect

    if expect.get("no_tool_calls"):
        if result.tool_calls:
            names = ", ".join(tc.name for tc in result.tool_calls)
            outcome.failures.append(f"expected no tool calls, got: {names}")
        default_finish = "stop"
    else:
        _check_tool_calls(case, expect["tool_calls"], result, outcome)
        default_finish = "tool_calls"

    expected_finish = expect.get("finish_reason", default_finish)
    if expected_finish != "any" and result.finish_reason != expected_finish:
        outcome.failures.append(
            f"expected finish_reason={expected_finish!r}, "
            f"got {result.finish_reason!r}"
        )

    content_expect = expect.get("content", {})
    if content_expect.get("nonempty") and not result.content.strip():
        outcome.failures.append("expected non-empty content")
    for needle in content_expect.get("contains", []):
        if needle not in result.content:
            outcome.failures.append(
                f"content missing {needle!r}: {truncate(result.content)}"
            )
    contains_any = content_expect.get("contains_any", [])
    if contains_any and not any(n in result.content for n in contains_any):
        outcome.failures.append(
            f"content contains none of {contains_any!r}: "
            f"{truncate(result.content)}"
        )

    return outcome


def _parse_arguments(tool_call: ToolCall) -> tuple[Any | None, str | None]:
    """Parse a tool call's arguments string; returns (value, error)."""
    try:
        return json.loads(tool_call.arguments), None
    except json.JSONDecodeError as exc:
        return None, (
            f"tool call {tool_call.name!r} arguments are not valid JSON "
            f"({exc.msg}): {truncate(tool_call.arguments)}"
        )


def _tool_schemas(request: dict[str, Any]) -> dict[str, Any]:
    schemas: dict[str, Any] = {}
    for tool in request.get("tools", []):
        function = tool.get("function", {})
        name = function.get("name")
        if name:
            schemas[name] = function.get("parameters")
    return schemas


def _check_tool_calls(
    case: BehaviorCase,
    spec: dict[str, Any],
    result: CompletionResult,
    outcome: CheckOutcome,
) -> None:
    calls = result.tool_calls
    minimum = spec.get("min", 1)
    maximum = spec.get("max")
    if len(calls) < minimum:
        outcome.failures.append(
            f"expected at least {minimum} tool call(s), got {len(calls)}; "
            f"{result.describe()}"
        )
        return
    if maximum is not None and len(calls) > maximum:
        outcome.failures.append(
            f"expected at most {maximum} tool call(s), got {len(calls)}: "
            + ", ".join(tc.name for tc in calls)
        )

    allowed = spec.get("names")
    if allowed is not None:
        for tc in calls:
            if tc.name not in allowed:
                outcome.failures.append(
                    f"unexpected tool name {tc.name!r} (allowed: {allowed})"
                )
    for required_name in spec.get("require_names", []):
        if not any(tc.name == required_name for tc in calls):
            outcome.failures.append(
                f"expected a call to {required_name!r}, got: "
                + ", ".join(tc.name for tc in calls)
            )

    # Arguments must always be valid JSON objects.
    parsed: list[Any | None] = []
    for tc in calls:
        value, error = _parse_arguments(tc)
        if error is not None:
            outcome.failures.append(error)
            parsed.append(None)
            continue
        if not isinstance(value, dict):
            outcome.failures.append(
                f"tool call {tc.name!r} arguments are not a JSON object: "
                f"{truncate(tc.arguments)}"
            )
            value = None
        parsed.append(value)

    if spec.get("args_match_schema", True):
        schemas = _tool_schemas(case.request)
        for tc, value in zip(calls, parsed):
            if value is None:
                continue
            schema = schemas.get(tc.name)
            if schema is None:
                continue
            try:
                Draft202012Validator(schema).validate(value)
            except SchemaError as exc:
                outcome.failures.append(
                    f"case tool schema itself is invalid: {truncate(str(exc))}"
                )
            except ValidationError as exc:
                outcome.failures.append(
                    f"tool call {tc.name!r} arguments do not match its schema: "
                    f"{truncate(exc.message)}; arguments={truncate(tc.arguments)}"
                )

    for check in spec.get("checks", []):
        error = _run_value_check(check, calls, parsed)
        if error is not None:
            outcome.failures.append(error)


def _resolve_path(value: Any, path: str) -> tuple[Any, bool]:
    current = value
    for segment in path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                return None, False
            current = current[segment]
        elif isinstance(current, list):
            try:
                current = current[int(segment)]
            except (ValueError, IndexError):
                return None, False
        else:
            return None, False
    return current, True


_JSON_TYPES = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool))
    or (isinstance(v, float) and v.is_integer()),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def _check_one_value(check: dict[str, Any], value: Any) -> str | None:
    """Apply the check's op to a resolved value; returns an error or None."""
    if "equals" in check:
        expected = check["equals"]
        if isinstance(expected, bool) or isinstance(value, bool):
            if value is not expected:
                return f"expected {expected!r}, got {value!r}"
        elif value != expected:
            return f"expected {expected!r}, got {value!r}"
        return None
    if "one_of" in check:
        if value not in check["one_of"]:
            return f"expected one of {check['one_of']!r}, got {value!r}"
        return None
    if "contains" in check:
        if not isinstance(value, str) or check["contains"] not in value:
            return f"expected substring {check['contains']!r} in {value!r}"
        return None
    if "contains_any" in check:
        needles = check["contains_any"]
        if not isinstance(value, str) or not any(n in value for n in needles):
            return f"expected any of {needles!r} in {value!r}"
        return None
    if "json_type" in check:
        type_name = check["json_type"]
        predicate = _JSON_TYPES.get(type_name)
        if predicate is None:
            return f"unknown json_type {type_name!r} in check"
        if not predicate(value):
            return f"expected JSON type {type_name}, got {value!r}"
        return None
    return f"check has no recognized op: {check!r}"


def _run_value_check(
    check: dict[str, Any],
    calls: list[ToolCall],
    parsed: list[Any | None],
) -> str | None:
    path = check.get("path")
    if path is None:
        return f"check is missing 'path': {check!r}"
    target = check.get("call", "any")

    if target == "any":
        candidates = list(enumerate(parsed))
    else:
        idx = int(target)
        if idx >= len(parsed):
            return f"check targets call {idx} but only {len(parsed)} call(s) present"
        candidates = [(idx, parsed[idx])]

    errors: list[str] = []
    for idx, value in candidates:
        if value is None:
            errors.append(f"call {idx}: arguments unparseable")
            continue
        resolved, found = _resolve_path(value, path)
        if not found:
            errors.append(
                f"call {idx} ({calls[idx].name}): path {path!r} not found in "
                f"{truncate(calls[idx].arguments)}"
            )
            continue
        error = _check_one_value(check, resolved)
        if error is None:
            return None
        errors.append(f"call {idx} ({calls[idx].name}) path {path!r}: {error}")

    return "; ".join(errors) or f"no tool call satisfied check {check!r}"
