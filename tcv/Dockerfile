FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Install locked dependencies first so source edits don't bust the layer cache.
WORKDIR /opt/tcv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . ./
RUN uv sync --frozen

ENV PATH="/opt/tcv/.venv/bin:$PATH"

# Default to the host's server and a mountable results directory. Both are
# overridable with the matching CLI flags.
ENV TCV_BASE_URL=http://localhost:8000/v1 \
    TCV_API_KEY=dummy

WORKDIR /results
ENTRYPOINT ["python", "/opt/tcv/run_tcv_eval.py", "--output-dir", "/results"]
CMD []
