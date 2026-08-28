"""OpenAI-compatible chat-completions helpers.

Normalizes non-streaming responses and streamed chunk sequences into a
single ``CompletionResult`` so expectation checks are mode-agnostic.

The streaming reassembly follows the OpenAI contract: ``tool_calls`` arrive
as deltas keyed by ``index``, ``function.name`` typically arrives first, and
``function.arguments`` is concatenated across chunks. Reassembly is where
serving-stack tool-call parsers most often break, which is exactly what
these evals are meant to catch — so the reassembly itself stays as lenient
as the spec allows and the expectation checks do the judging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def get_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def truncate(value: str, limit: int = 500) -> str:
    value = value.replace("\n", "\\n")
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


@dataclass
class ToolCall:
    name: str
    arguments: str
    call_id: str | None = None


@dataclass
class CompletionResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    chunk_count: int = 0

    def describe(self) -> str:
        calls = ", ".join(
            f"{tc.name}({truncate(tc.arguments, 200)})" for tc in self.tool_calls
        )
        return (
            f"finish_reason={self.finish_reason} tool_calls=[{calls}] "
            f"content={truncate(self.content, 200)}"
        )


def from_completion(completion: Any) -> CompletionResult:
    """Normalize a non-streaming chat completion."""
    choices = get_field(completion, "choices") or []
    if not choices:
        return CompletionResult()
    choice = choices[0]
    message = get_field(choice, "message")
    result = CompletionResult(finish_reason=get_field(choice, "finish_reason"))
    if message is None:
        return result
    content = get_field(message, "content")
    if isinstance(content, str):
        result.content = content
    for tool_call in get_field(message, "tool_calls") or []:
        function = get_field(tool_call, "function")
        name = get_field(function, "name")
        arguments = get_field(function, "arguments")
        result.tool_calls.append(
            ToolCall(
                name=str(name) if name is not None else "",
                arguments=arguments if isinstance(arguments, str) else "",
                call_id=get_field(tool_call, "id"),
            )
        )
    return result


def from_stream(stream: Any) -> CompletionResult:
    """Reassemble a streamed chat completion into a CompletionResult."""
    by_index: dict[int, dict[str, Any]] = {}
    result = CompletionResult()

    for chunk in stream:
        result.chunk_count += 1
        choices = get_field(chunk, "choices") or []
        if not choices:
            continue
        choice = choices[0]
        finish = get_field(choice, "finish_reason")
        if finish:
            result.finish_reason = finish
        delta = get_field(choice, "delta")
        if delta is None:
            continue
        content_delta = get_field(delta, "content")
        if isinstance(content_delta, str):
            result.content += content_delta
        for tc_delta in get_field(delta, "tool_calls") or []:
            idx = get_field(tc_delta, "index")
            entry = by_index.setdefault(
                int(idx) if idx is not None else 0,
                {"name": "", "arguments": "", "call_id": None},
            )
            call_id = get_field(tc_delta, "id")
            if call_id:
                entry["call_id"] = call_id
            function = get_field(tc_delta, "function")
            if function is None:
                continue
            name_delta = get_field(function, "name")
            if isinstance(name_delta, str):
                entry["name"] += name_delta
            args_delta = get_field(function, "arguments")
            if isinstance(args_delta, str):
                entry["arguments"] += args_delta

    for idx in sorted(by_index):
        entry = by_index[idx]
        result.tool_calls.append(
            ToolCall(
                name=entry["name"],
                arguments=entry["arguments"],
                call_id=entry["call_id"],
            )
        )
    return result


def request_completion(
    client: Any,
    model: str,
    request: dict[str, Any],
    *,
    stream: bool,
    max_tokens: int,
    extra_body: dict[str, Any] | None = None,
) -> CompletionResult:
    """Send one chat completion built from a case ``request`` dict."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": request["messages"],
        "max_tokens": request.get("max_tokens", max_tokens),
    }
    for key in ("tools", "tool_choice", "parallel_tool_calls", "temperature"):
        if key in request:
            kwargs[key] = request[key]
    if extra_body:
        kwargs["extra_body"] = extra_body
    if stream:
        return from_stream(client.chat.completions.create(stream=True, **kwargs))
    return from_completion(client.chat.completions.create(**kwargs))
