"""tool_choice contract tests, gated by the model-family profile.

    "auto"     model decides — the prompt content drives the choice
    "none"     no tool call, even if the prompt encourages one
    "required" at least one tool call, even if the prompt discourages one
    named      {"type": "function", "function": {"name": ...}} forces that
               specific tool (skipped for profiles that don't support it)

Server-side rejection behavior for invalid values differs across serving
stacks and is deliberately not asserted here — these tests only pin down
the positive contract.
"""

from __future__ import annotations

import json

import openai
import pytest

from tcv.openai_compat import from_completion
from tcv.profiles import Profile


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current local time for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _create(client: openai.Client, model: str, extra_body: dict, **kwargs):
    if extra_body:
        kwargs["extra_body"] = extra_body
    return from_completion(
        client.chat.completions.create(model=model, max_tokens=2048, **kwargs)
    )


@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_auto_may_call(client: openai.Client, model: str, extra_body: dict):
    """With auto, a prompt that needs a tool (and asks to use one) leads to a
    tool call."""
    result = _create(
        client,
        model,
        extra_body,
        messages=[
            {
                "role": "user",
                "content": "What's the weather in Beijing right now? "
                "Look it up — you should use a tool.",
            }
        ],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
    )
    assert result.tool_calls, f"expected a tool call, got: {result.describe()}"
    assert result.finish_reason == "tool_calls", result.describe()


@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_auto_may_not_call(client: openai.Client, model: str, extra_body: dict):
    """With auto, a prompt that needs no tool (and asks not to use one) leads
    to a plain text answer."""
    result = _create(
        client,
        model,
        extra_body,
        messages=[
            {
                "role": "user",
                "content": "Hi! Briefly introduce yourself. "
                "Do not use any tools.",
            }
        ],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
    )
    assert not result.tool_calls, (
        f"tool_choice='auto' should not call a tool here: {result.describe()}"
    )
    assert result.finish_reason == "stop", result.describe()
    assert result.content.strip(), f"expected content, got: {result.describe()}"


def test_required_forces_call(
    client: openai.Client, model: str, extra_body: dict, profile: Profile
):
    if not profile.supports_tool_choice_required:
        pytest.skip(f"profile {profile.name} does not support tool_choice=required")
    result = _create(
        client,
        model,
        extra_body,
        messages=[{"role": "user", "content": "Briefly: how is the weather in Beijing?"}],
        tools=[WEATHER_TOOL],
        tool_choice="required",
    )
    assert result.tool_calls, (
        f"tool_choice='required' should force a tool call: {result.describe()}"
    )
    args = json.loads(result.tool_calls[0].arguments)
    assert isinstance(args, dict), f"arguments not an object: {args!r}"


def test_none_forbids_call(client: openai.Client, model: str, extra_body: dict):
    result = _create(
        client,
        model,
        extra_body,
        messages=[{"role": "user", "content": "Please look up the weather in Beijing."}],
        tools=[WEATHER_TOOL],
        tool_choice="none",
    )
    assert not result.tool_calls, (
        f"tool_choice='none' should not call a tool: {result.describe()}"
    )
    assert result.finish_reason == "stop", result.describe()


def test_named_function_forces_that_tool(
    client: openai.Client, model: str, extra_body: dict, profile: Profile
):
    """The named form must call exactly the named tool — even when the
    prompt pulls toward the other tool."""
    if not profile.supports_tool_choice_named:
        pytest.skip(f"profile {profile.name} does not support named tool_choice")
    result = _create(
        client,
        model,
        extra_body,
        messages=[{"role": "user", "content": "What time is it in Tokyo?"}],
        tools=[WEATHER_TOOL, TIME_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    assert result.tool_calls, (
        f"named tool_choice should force a tool call: {result.describe()}"
    )
    names = {tc.name for tc in result.tool_calls}
    assert names == {"get_weather"}, (
        f"named tool_choice must call only get_weather, got: {names}"
    )
    args = json.loads(result.tool_calls[0].arguments)
    assert isinstance(args, dict) and "city" in args, (
        f"expected a city argument, got: {result.tool_calls[0].arguments!r}"
    )
