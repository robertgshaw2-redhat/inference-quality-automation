FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN uv pip install --system \
    "inspect-ai>=0.3.258" \
    "inspect-evals[bfcl]>=0.18.0,<1" \
    "openai>=1.0"

# Pre-download the BFCL multi-turn datasets (question, possible_answer,
# function docs) into the image cache so eval runs don't need GitHub access.
RUN python -c "\
from inspect_evals.bfcl import bfcl; \
from inspect_evals.bfcl.utils.task_categories import V3_CATEGORIES; \
[bfcl(categories=[c]) for c in sorted(V3_CATEGORIES)]"

WORKDIR /results
COPY run_bfcl_inspect.py /opt/bfcl-inspect/run_bfcl_inspect.py

ENTRYPOINT ["python", "/opt/bfcl-inspect/run_bfcl_inspect.py", "--output-dir", "/results"]
CMD []
