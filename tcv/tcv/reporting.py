"""xdist-safe JSON report writing for the pytest suites.

Pattern borrowed from the KVV tool_call_json_schema conftest: each xdist
worker appends records to its own JSONL sidecar file; the master process
merges them in ``pytest_sessionfinish`` and writes one JSON report with a
summary. pytest-rerunfailures may record the same item several times — the
last record per key wins.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_RESULTS_SUFFIX = ".results.jsonl"
MAX_MESSAGE_LEN = 2000


def is_worker() -> bool:
    return bool(os.environ.get("PYTEST_XDIST_WORKER"))


def _worker_results_path(report_path: Path) -> Path:
    worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    return report_path.parent / f"{report_path.name}{_RESULTS_SUFFIX}.{worker}"


def clean_stale_results(report_path: Path) -> None:
    """Master process removes leftover per-worker files before a run."""
    if is_worker():
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for p in report_path.parent.glob(f"{report_path.name}{_RESULTS_SUFFIX}.*"):
        p.unlink(missing_ok=True)


def append_record(report_path: Path, record: dict[str, Any]) -> None:
    path = _worker_results_path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_worker_results(report_path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for p in sorted(report_path.parent.glob(f"{report_path.name}{_RESULTS_SUFFIX}.*")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        p.unlink(missing_ok=True)
    return results


def finalize_report(
    report_path: Path,
    *,
    meta: dict[str, Any],
    key_fields: tuple[str, ...],
) -> None:
    """Merge worker records, dedupe by *key_fields*, and write the report."""
    if is_worker():
        return
    raw = _load_worker_results(report_path)
    if not raw:
        return

    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in raw:
        by_key[tuple(record.get(k) for k in key_fields)] = record
    results = sorted(
        by_key.values(), key=lambda r: tuple(str(r.get(k)) for k in key_fields)
    )

    statuses: dict[str, int] = {}
    by_mode: dict[str, dict[str, int]] = {}
    for r in results:
        status = r.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        mode = r.get("mode")
        if mode is not None:
            mode_stats = by_mode.setdefault(mode, {})
            mode_stats[status] = mode_stats.get(status, 0) + 1

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        **meta,
        "summary": {
            "total": len(results),
            "by_status": dict(sorted(statuses.items())),
            "by_mode": {m: dict(sorted(s.items())) for m, s in sorted(by_mode.items())},
        },
        "results": results,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
