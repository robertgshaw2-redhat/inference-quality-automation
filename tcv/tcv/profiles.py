"""Model-family profiles.

A profile captures the per-model-family knobs that an OpenAI-compatible
request cannot express portably:

- how to toggle "thinking" (the chat-template kwarg name differs per family)
- which tool_choice forms the family's serving stack is expected to support

Add a new family by adding an entry to PROFILES. Everything else in the
suites is driven off the profile object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Profile:
    name: str
    # chat_template_kwargs key that toggles thinking, or None if the family
    # has no such kwarg (nothing is sent either way).
    thinking_kwarg: str | None
    # Whether tool_choice="required" is expected to work on this family's
    # recommended serving stack.
    supports_tool_choice_required: bool = True
    # Whether the named-function form of tool_choice is expected to work.
    supports_tool_choice_named: bool = True


PROFILES: dict[str, Profile] = {
    # GLM-4.5 / GLM-4.6 served by vLLM/SGLang: thinking is on by default and
    # toggled with chat_template_kwargs={"enable_thinking": bool}.
    "glm": Profile(name="glm", thinking_kwarg="enable_thinking"),
    # Qwen3 uses the same kwarg name as GLM.
    "qwen3": Profile(name="qwen3", thinking_kwarg="enable_thinking"),
    # Kimi K2/K3 open-source serving uses chat_template_kwargs={"thinking": bool}
    # and does not support named-function tool_choice.
    "kimi": Profile(
        name="kimi",
        thinking_kwarg="thinking",
        supports_tool_choice_named=False,
    ),
    # DeepSeek V3.1+ hybrid thinking toggle.
    "deepseek": Profile(name="deepseek", thinking_kwarg="thinking"),
    # Unknown family: never send thinking kwargs, assume full tool_choice.
    "generic": Profile(name="generic", thinking_kwarg=None),
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown profile {name!r}; choose from {', '.join(sorted(PROFILES))}"
        ) from None


def thinking_extra_body(profile: Profile, thinking: str) -> dict[str, Any]:
    """Build the request extra_body for a thinking setting.

    ``thinking`` is one of:
        "default"  send nothing; the server/template default applies
        "on"       explicitly enable thinking
        "off"      explicitly disable thinking
    """
    if thinking == "default" or profile.thinking_kwarg is None:
        return {}
    if thinking not in ("on", "off"):
        raise ValueError(f"invalid thinking setting: {thinking!r}")
    return {"chat_template_kwargs": {profile.thinking_kwarg: thinking == "on"}}
