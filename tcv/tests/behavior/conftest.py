"""JSON report for the behavior suite (xdist-safe; see tcv/reporting.py)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tcv import reporting


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--behavior-report",
        default=os.environ.get("TCV_BEHAVIOR_REPORT", "behavior-report.json"),
        help="Write behavior-suite JSON report to this path "
        "(default: behavior-report.json)",
    )


def _report_path(config: pytest.Config) -> Path:
    return Path(config.getoption("behavior_report")).resolve()


def _item_case_run(item: pytest.Item):
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return None
    return callspec.params.get("case_run")


def pytest_configure(config: pytest.Config) -> None:
    reporting.clean_stale_results(_report_path(config))


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    if call.when != "call" or "case_run" not in getattr(item, "fixturenames", []):
        return
    params = _item_case_run(item)
    if params is None:
        return

    case, mode = params
    report: pytest.TestReport = outcome.get_result()
    message = ""
    if report.outcome != "passed" and report.longrepr is not None:
        text = str(report.longrepr)
        if len(text) > reporting.MAX_MESSAGE_LEN:
            text = text[: reporting.MAX_MESSAGE_LEN - 3] + "..."
        message = text

    reporting.append_record(
        _report_path(item.config),
        {
            "case_id": case.id,
            "mode": mode,
            "source": case.source,
            "tags": list(case.tags),
            "description": case.description,
            "status": report.outcome,
            "message": message,
        },
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    reporting.finalize_report(
        _report_path(config),
        meta={
            "suite": "behavior",
            "model": config.getoption("model"),
            "base_url": config.getoption("base_url"),
            "profile": config.getoption("profile"),
            "thinking": config.getoption("thinking"),
        },
        key_fields=("case_id", "mode"),
    )
