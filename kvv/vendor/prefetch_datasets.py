#!/usr/bin/env python3
"""Populate benchmark dataset caches, for use at image build time.

Constructing each Task downloads its dataset into the cache steered by
XDG_CACHE_HOME / HF_HOME (aime2025, ocrbench via inspect_ai's hf_dataset;
mmmu via the datasets library; bfcl into inspect_evals' cache), so eval
runs need no Hugging Face or GitHub access.

Usage: python prefetch_datasets.py [aime2025|ocrbench|mmmu|bfcl ...]
"""

import sys


def prefetch(bench: str) -> None:
    if bench == "aime2025":
        from aime2025 import aime2025

        aime2025()
    elif bench == "ocrbench":
        from ocr_bench import ocrbench

        ocrbench()
    elif bench == "mmmu":
        from mmmu_pro_vision import mmmu_pro_10c

        mmmu_pro_10c()
    elif bench == "bfcl":
        from bfcl_multi_turn import bfcl_multi_turn

        bfcl_multi_turn()
    else:
        raise SystemExit(
            f"unknown benchmark {bench!r}; expected aime2025, ocrbench, mmmu, or bfcl"
        )


def main() -> None:
    benchmarks = sys.argv[1:]
    if not benchmarks:
        raise SystemExit("usage: prefetch_datasets.py <benchmark> [...]")
    for bench in benchmarks:
        print(f"=== prefetching {bench} ===")
        prefetch(bench)


if __name__ == "__main__":
    main()
