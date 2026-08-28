FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Install locked dependencies first so source edits don't bust the layer cache.
WORKDIR /opt/kvv/vendor
COPY vendor/pyproject.toml vendor/uv.lock ./
RUN uv sync --frozen --no-install-project

ENV PATH="/opt/kvv/vendor/.venv/bin:$PATH"

# Dataset caches live in the image so evals need no Hugging Face access at run
# time. XDG_CACHE_HOME steers inspect_ai's hf_dataset cache (aime2025,
# ocrbench) and inspect_evals' cache (bfcl); HF_HOME steers the datasets
# cache (mmmu).
ENV XDG_CACHE_HOME=/opt/kvv/cache \
    HF_HOME=/opt/kvv/cache/huggingface

# Only the modules that define the datasets, so edits elsewhere don't force a
# re-download of the (large) cache layer below.
COPY vendor/aime2025.py vendor/ocr_bench.py vendor/mmmu_pro_vision.py \
     vendor/bfcl_multi_turn.py vendor/prefetch_datasets.py ./

# Space-separated subset of "aime2025 ocrbench mmmu bfcl". Trimming this
# shrinks the image (mmmu ~700MB, ocrbench ~70MB, bfcl ~15MB, aime2025
# negligible) at the cost of only being able to run the benchmarks that
# remain cached.
ARG PREFETCH_BENCHMARKS="aime2025 ocrbench mmmu bfcl"
RUN python prefetch_datasets.py ${PREFETCH_BENCHMARKS}

COPY vendor/ ./
RUN uv sync --frozen

# kimi_model writes its retry log next to the source; keep the caches and that
# log directory writable so the image also runs under an arbitrary UID.
RUN mkdir -p /opt/kvv/vendor/logs /results \
    && chmod -R a+rwX /opt/kvv/cache /opt/kvv/vendor/logs /results

# Point at the host's server by default; override with -e at run time.
ENV KIMI_BASE_URL=http://localhost:8000/v1 \
    KIMI_API_KEY=dummy \
    INSPECT_LOG_DIR=/results/logs \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

WORKDIR /results
ENTRYPOINT ["python", "/opt/kvv/vendor/eval.py"]
CMD []
