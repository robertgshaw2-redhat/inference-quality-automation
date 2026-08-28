FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

# soundfile is an undeclared transitive requirement: importing BFCL's model
# config pulls in qwen_agent, which imports it at module load.
RUN uv pip install --system "bfcl-eval>=2025.10.20.1,<2026" soundfile --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /results
COPY run_bfcl_eval.py /opt/bfcl/run_bfcl_eval.py

ENTRYPOINT ["python", "/opt/bfcl/run_bfcl_eval.py", "--output-dir", "/results"]
CMD []
