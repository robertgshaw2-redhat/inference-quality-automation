"""Materialize benchmark datasets into the image so evals can run offline.

Run at image build time. Warms the caches the benchmarks read from:
`inspect_ai`'s `hf_dataset` disk cache (aime2025, gsm8k, ocrbench), the
`datasets` cache (mmmu, which calls `load_dataset` directly), and
`inspect_evals`' cache (bfcl).
"""

import argparse
import sys

BENCHMARKS = ("aime2025", "ocrbench", "mmmu", "bfcl", "gsm8k", "gpqa")


def prefetch_aime2025() -> int:
    from aime2025 import aime2025

    return len(aime2025().dataset)


def prefetch_ocrbench() -> int:
    from ocr_bench import ocrbench

    return len(ocrbench().dataset)


def prefetch_mmmu() -> int:
    from datasets import load_dataset

    from mmmu_pro_vision import MMMU_PRO_10c_DATASET, MMMU_PRO_10c_SUBSET

    # Only warm the `datasets` cache; converting rows to samples decodes every
    # image and buys nothing that is not redone at eval time anyway.
    return len(load_dataset(MMMU_PRO_10c_DATASET, MMMU_PRO_10c_SUBSET, split="test"))


def prefetch_bfcl() -> int:
    from bfcl_multi_turn import bfcl_multi_turn

    return len(bfcl_multi_turn().dataset)


def prefetch_gsm8k() -> int:
    from gsm8k import gsm8k

    return len(gsm8k().dataset)


def prefetch_gpqa() -> int:
    from gpqa_diamond import gpqa_diamond

    return len(gpqa_diamond().dataset)


PREFETCHERS = {
    "aime2025": prefetch_aime2025,
    "ocrbench": prefetch_ocrbench,
    "mmmu": prefetch_mmmu,
    "bfcl": prefetch_bfcl,
    "gsm8k": prefetch_gsm8k,
    "gpqa": prefetch_gpqa,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmarks",
        nargs="*",
        default=list(BENCHMARKS),
        choices=list(BENCHMARKS),
        help="Benchmarks to prefetch (default: all)",
    )
    args = parser.parse_args()

    for bench in args.benchmarks or BENCHMARKS:
        print(f"--- prefetching {bench} ---", flush=True)
        count = PREFETCHERS[bench]()
        print(f"--- {bench}: {count} samples cached ---", flush=True)

    print("all datasets cached", file=sys.stderr)


if __name__ == "__main__":
    main()
