FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

# soundfile is an undeclared transitive requirement: importing BFCL's model
# config pulls in qwen_agent, which imports it at module load.
RUN uv pip install --system "bfcl-eval>=2025.10.20.1,<2026" soundfile --extra-index-url https://download.pytorch.org/whl/cpu

ENV HOME=/tmp

COPY run_bfcl_eval.py /opt/bfcl/run_bfcl_eval.py

# OpenShift runs as an arbitrary UID in group 0: make the results dir group-0
# writable with group perms mirroring owner perms.
RUN mkdir -p /results \
    && chgrp -R 0 /results \
    && chmod -R g=u /results

WORKDIR /results

USER 1001

ENTRYPOINT ["python", "/opt/bfcl/run_bfcl_eval.py", "--output-dir", "/results"]
CMD []
