FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Install locked dependencies first so source edits don't bust the layer cache.
WORKDIR /opt/kvv/vendor
COPY vendor/pyproject.toml vendor/uv.lock ./
RUN uv sync --frozen --no-install-project

COPY vendor/ ./
RUN uv sync --frozen

ENV PATH="/opt/kvv/vendor/.venv/bin:$PATH"

COPY run_kvv_eval.py /opt/kvv/run_kvv_eval.py

# Default to the host's server and a mountable results directory. Both are
# overridable with the matching CLI flags.
ENV KIMI_BASE_URL=http://localhost:8000/v1 \
    KIMI_API_KEY=dummy

WORKDIR /results
ENTRYPOINT ["python", "/opt/kvv/run_kvv_eval.py", "--output-dir", "/results"]
CMD []
