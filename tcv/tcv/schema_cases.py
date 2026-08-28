"""Schema-fuzz cases: tool-call arguments validated against JSON Schemas.

Each walle-valid MFJS schema (testdata/walle_validator_cases) is sent as a
tool's ``parameters``, the model is forced to call the tool, and the
returned ``function.arguments`` are validated against the schema with
jsonschema — in both non-streaming and streaming modes.

Case loading, classification, and schema wrapping are adapted from
MoonshotAI/kimi-vendor-verifier (MIT), generalized to be model-agnostic:
the Kimi-specific thinking parameters are replaced by profile-driven
``extra_body`` built in the test fixtures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from tcv.openai_compat import CompletionResult, truncate


TOOL_NAME = "tcv_schema_case"
TOOL_KEYWORDS = (
    "tool",
    "function",
    "tool_call",
    "function_call",
    "arguments",
    "parameters",
)
REQUEST_MODES = ("non-stream", "stream")

INSTRUCTION_PROMPT = (
    f"Call the {TOOL_NAME} tool exactly once with minimum "
    "runtime arguments that satisfy its parameter schema, "
    "try your best to create the arguments. "
    "Do not copy or describe the JSON Schema itself. Do not "
    "include schema keywords like type, properties, required, "
    "or additionalProperties unless the schema explicitly "
    "requires them as argument property names. If the schema "
    "defines a top-level value argument, provide the minimal "
    "valid value for it. Always include every required "
    "property. Respect minItems, minProperties, enum, const, minimum, and "
    "minLength constraints. Prefer empty arrays and empty "
    "objects only when those constraints allow them. Do not "
    "answer with plain text."
)


@dataclass(frozen=True)
class ValidatorCase:
    suite: str
    line: int
    schema_text: str


def load_cases(case_dir: Path) -> list[ValidatorCase]:
    if not case_dir.is_dir():
        raise FileNotFoundError(f"case directory not found: {case_dir}")

    cases: list[ValidatorCase] = []
    for suite_path in sorted(p for p in case_dir.iterdir() if p.is_dir()):
        valid_path = suite_path / "valid.jsonl"
        with valid_path.open(encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                schema_text = raw_line.strip()
                if not schema_text:
                    continue
                cases.append(
                    ValidatorCase(
                        suite=suite_path.name,
                        line=line_number,
                        schema_text=schema_text,
                    )
                )
    return cases


def has_non_finite_number(value: Any) -> bool:
    if isinstance(value, float):
        return value != value or value in (float("inf"), float("-inf"))
    if isinstance(value, list):
        return any(has_non_finite_number(item) for item in value)
    if isinstance(value, dict):
        return any(has_non_finite_number(item) for item in value.values())
    return False


def parse_for_transport(case: ValidatorCase) -> tuple[bool, Any | None, str]:
    try:
        schema = json.loads(case.schema_text)
    except json.JSONDecodeError as exc:
        return False, None, f"schema text is not valid JSON: {exc.msg}"
    if has_non_finite_number(schema):
        return False, None, "schema contains non-finite numeric value"
    return True, schema, ""


def schema_text_has_tool_keyword(schema: Any) -> bool:
    text = json.dumps(schema, ensure_ascii=False).lower()
    return any(keyword in text for keyword in TOOL_KEYWORDS)


def is_object_parameter_schema(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    schema_type = schema.get("type")
    return (
        schema_type == "object"
        or (isinstance(schema_type, list) and "object" in schema_type)
    ) and isinstance(schema.get("properties"), dict)


def rewrite_root_refs(obj: Any, target_ref: str) -> Any:
    """Rewrite JSON Schema root self-references to *target_ref*."""
    if isinstance(obj, dict):
        rewritten: dict[str, Any] = {}
        for key, value in obj.items():
            if key == "$ref" and value == "#":
                rewritten[key] = target_ref
            else:
                rewritten[key] = rewrite_root_refs(value, target_ref)
        return rewritten
    if isinstance(obj, list):
        return [rewrite_root_refs(item, target_ref) for item in obj]
    return obj


def wrap_schema_as_parameter_property(schema: Any) -> Any:
    """Use the case schema as one tool argument property schema."""
    if not isinstance(schema, dict):
        return schema

    wrapped: dict[str, Any] = {
        "type": "object",
        "required": ["value"],
        "additionalProperties": False,
    }

    if "#" in _collect_refs(schema):
        defs = dict(schema.get("$defs", {}))
        def_name = "__case_schema"
        while def_name in defs:
            def_name = f"_{def_name}"
        target_ref = f"#/$defs/{def_name}"
        rewritten = rewrite_root_refs(schema, target_ref)
        defs.update(rewritten.get("$defs", {}))
        defs[def_name] = {
            k: v for k, v in rewritten.items() if k not in ("$defs", "$id")
        }
        wrapped["properties"] = {"value": {"$ref": target_ref}}
        wrapped["$defs"] = defs
    else:
        wrapped["properties"] = {
            "value": {k: v for k, v in schema.items() if k not in ("$defs", "$id")}
        }
        if "$defs" in schema:
            wrapped["$defs"] = schema["$defs"]

    if "$id" in schema:
        wrapped["$id"] = schema["$id"]
    return wrapped


def _collect_refs(obj: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(obj, dict):
        if "$ref" in obj:
            refs.append(obj["$ref"])
        for v in obj.values():
            refs.extend(_collect_refs(v))
    elif isinstance(obj, list):
        for item in obj:
            refs.extend(_collect_refs(item))
    return refs


def _is_primitive_type(t: Any) -> bool:
    if isinstance(t, str):
        return t in ("string", "number", "integer", "boolean", "null")
    if isinstance(t, list):
        return any(_is_primitive_type(x) for x in t)
    return False


def _has_non_recursive_option(obj: Any) -> bool:
    """Return True if *obj* allows a value that does not recurse."""
    if isinstance(obj, dict):
        if _is_primitive_type(obj.get("type")):
            return True
        if "anyOf" in obj:
            return any(_has_non_recursive_option(s) for s in obj["anyOf"])
        if "oneOf" in obj:
            return any(_has_non_recursive_option(s) for s in obj["oneOf"])
        if "enum" in obj:
            return True
        for v in obj.values():
            if _has_non_recursive_option(v):
                return True
    elif isinstance(obj, list):
        return any(_has_non_recursive_option(item) for item in obj)
    return False


def has_recursive_ref_without_termination(schema: Any) -> bool:
    """Detect recursive $ref that cannot terminate (no escape + required)."""
    if not isinstance(schema, dict):
        return False
    defs = schema.get("$defs", {})
    for ref in _collect_refs(schema):
        if not ref.startswith("#/$defs/"):
            continue
        def_name = ref.split("/")[-1]
        if def_name not in defs:
            continue
        def_schema = defs[def_name]
        for inner_ref in _collect_refs(def_schema):
            if inner_ref != ref:
                continue
            if _has_non_recursive_option(def_schema):
                continue
            required = set(def_schema.get("required", []))
            props = def_schema.get("properties", {})
            for prop_name, prop_schema in props.items():
                if ref in _collect_refs(prop_schema) and prop_name in required:
                    return True
    return False


def has_exotic_property_keys(schema: Any) -> bool:
    """Detect property keys known to confuse tool-call decoders.

    Leading/trailing whitespace, empty-string keys, and literal escape
    sequences in keys are mishandled by several decoders, so those cases are
    skipped rather than counted against the model.
    """
    if not isinstance(schema, dict):
        return False
    for key in schema.get("properties", {}).keys():
        if key != key.strip():
            return True
        if key == "":
            return True
        if any(seq in key for seq in (r"\n", r"\t", r"\r", r"\b", r"\f")):
            return True
    return False


def strip_keyword_recursive(obj: Any, keyword: str) -> Any:
    """Recursively remove *keyword* from every dict in *obj*."""
    if isinstance(obj, dict):
        return {
            k: strip_keyword_recursive(v, keyword)
            for k, v in obj.items()
            if k != keyword
        }
    if isinstance(obj, list):
        return [strip_keyword_recursive(item, keyword) for item in obj]
    return obj


def schema_shape(schema: Any) -> str:
    if not isinstance(schema, dict):
        return type(schema).__name__
    if not schema:
        return "empty_parameter_schema"
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return "union_parameter_schema"
    if isinstance(schema_type, str):
        return f"{schema_type}_parameter_schema"
    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in schema:
            return f"{keyword}_parameter_schema"
    if "$ref" in schema:
        return "ref_parameter_schema"
    return "schema_parameter_schema"


def classify_case(case: ValidatorCase) -> tuple[Any | None, str | None]:
    transportable, schema, reason = parse_for_transport(case)
    if not transportable:
        return None, f"unsupported_by_transport: {reason}"

    if isinstance(schema, dict):
        if schema.get("type") == "string":
            min_len = schema.get("minLength")
            if min_len is not None and min_len > 1000:
                return None, "skipped_extreme_minlength_not_supported"
        if has_exotic_property_keys(schema):
            return None, "skipped_exotic_property_keys"
        if has_recursive_ref_without_termination(schema):
            return None, "skipped_recursive_ref"

    if schema_text_has_tool_keyword(schema):
        selection_reason = "explicit_tool_keyword"
    elif is_object_parameter_schema(schema):
        selection_reason = "object_parameter_schema"
    else:
        selection_reason = schema_shape(schema)

    schema = wrap_schema_as_parameter_property(schema)
    # ``default`` triggers decoder instability on several stacks.
    schema = strip_keyword_recursive(schema, "default")

    return schema, selection_reason


def select_cases(
    cases: list[ValidatorCase],
    *,
    max_cases: int | None,
) -> list[tuple[ValidatorCase, Any, str]]:
    selected: list[tuple[ValidatorCase, Any, str]] = []
    for case in cases:
        schema, reason = classify_case(case)
        if schema is None:
            continue
        selected.append((case, schema, reason))
        if max_cases is not None and len(selected) >= max_cases:
            break
    return selected


def build_request(schema: Any) -> dict[str, Any]:
    """Build the behavior-style request dict for one schema case."""
    return {
        "messages": [{"role": "user", "content": INSTRUCTION_PROMPT}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "description": (
                        "Submit minimal JSON arguments that validate against "
                        "this JSON Schema."
                    ),
                    "parameters": schema,
                    "strict": True,
                },
            }
        ],
        "tool_choice": "required",
    }


def extract_case_arguments(result: CompletionResult) -> tuple[str | None, str]:
    """Find the TOOL_NAME call in a normalized result."""
    if not result.tool_calls:
        return None, f"response had no tool_calls; {result.describe()}"
    for tc in result.tool_calls:
        if tc.name == TOOL_NAME:
            return tc.arguments, f"tool call {TOOL_NAME} returned arguments"
    names = ", ".join(tc.name for tc in result.tool_calls)
    return None, f"response did not include {TOOL_NAME} tool call; tool_calls={names}"


def validate_arguments(schema: Any, arguments: str | None) -> tuple[bool, str]:
    if arguments is None:
        return False, "tool call arguments are missing"

    try:
        instance = json.loads(arguments)
    except json.JSONDecodeError as exc:
        return (
            False,
            "tool call arguments are not valid JSON: "
            f"{exc.msg}; arguments={truncate(arguments)}",
        )

    try:
        Draft202012Validator(schema).validate(instance)
    except SchemaError as exc:
        return False, f"local jsonschema rejected schema: {truncate(str(exc))}"
    except ValidationError as exc:
        return (
            False,
            "tool call arguments do not match schema: "
            f"{truncate(exc.message)}; arguments={truncate(arguments)}",
        )

    return True, f"tool call arguments matched schema; arguments={truncate(arguments)}"
