"""Shared fixtures for all TCV suites."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import openai
import pytest

from tcv.profiles import PROFILES, Profile, get_profile, thinking_extra_body


ROOT = Path(__file__).parent.parent


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--base-url",
        default=os.environ.get("TCV_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible base URL (default: $TCV_BASE_URL or "
        "http://localhost:8000/v1)",
    )
    parser.addoption(
        "--api-key",
        default=os.environ.get("TCV_API_KEY", "dummy"),
        help="API key (default: $TCV_API_KEY or 'dummy')",
    )
    parser.addoption(
        "--model",
        default=os.environ.get("MODEL_NAME", ""),
        help="Model name to test (default: $MODEL_NAME)",
    )
    parser.addoption(
        "--profile",
        choices=sorted(PROFILES),
        default=os.environ.get("TCV_PROFILE", "glm"),
        help="Model-family profile (default: $TCV_PROFILE or glm)",
    )
    parser.addoption(
        "--thinking",
        choices=["default", "on", "off"],
        default=os.environ.get("TCV_THINKING", "default"),
        help="Thinking mode: 'default' sends nothing, 'on'/'off' send the "
        "profile's chat_template_kwargs toggle (default: default)",
    )
    parser.addoption(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("TCV_MAX_TOKENS", "2048")),
        help="Default max output tokens per request (default: 2048)",
    )
    parser.addoption(
        "--cases-dir",
        default=str(ROOT / "cases"),
        help="Directory of behavior case *.jsonl files (default: cases/)",
    )
    parser.addoption(
        "--tags",
        default="",
        help="Comma-separated tag filter for behavior cases (default: all)",
    )
    parser.addoption(
        "--case-filter",
        default="",
        help="Substring filter on behavior case ids (default: all)",
    )
    parser.addoption(
        "--schema-case-dir",
        default=str(ROOT / "testdata" / "walle_validator_cases" / "validator_cases"),
        help="walle validator_cases directory for the schema suite",
    )
    parser.addoption(
        "--max-cases",
        type=int,
        default=None,
        help="Cap on schema-suite cases (default: all)",
    )


@pytest.fixture(scope="session")
def profile(pytestconfig: pytest.Config) -> Profile:
    return get_profile(pytestconfig.getoption("profile"))


@pytest.fixture(scope="session")
def extra_body(pytestconfig: pytest.Config, profile: Profile) -> dict[str, Any]:
    return thinking_extra_body(profile, pytestconfig.getoption("thinking"))


@pytest.fixture(scope="session")
def model(pytestconfig: pytest.Config) -> str:
    name = pytestconfig.getoption("model")
    if not name:
        pytest.fail("no model given: pass --model or set MODEL_NAME")
    return name


@pytest.fixture(scope="session")
def default_max_tokens(pytestconfig: pytest.Config) -> int:
    return pytestconfig.getoption("max_tokens")


@pytest.fixture(scope="function")
def client(pytestconfig: pytest.Config) -> openai.Client:
    return openai.Client(
        api_key=pytestconfig.getoption("api_key"),
        base_url=str(pytestconfig.getoption("base_url")).rstrip("/"),
        timeout=120,
    )
