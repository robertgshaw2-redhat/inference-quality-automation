FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

# openai is an optional inspect-ai dependency, required by its
# OpenAI-compatible model provider.
RUN uv pip install --system "inspect-ai>=0.3.261,<0.4" "datasets>=3,<6" "openai>=3.1,<4"

# Bake the GSM8K dataset into the image so runs don't need Hugging Face
# access; HF_HUB_OFFLINE keeps runtime loads on the baked cache.
ENV HF_HOME=/opt/hf-cache
RUN python -c "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main')"
ENV HF_HUB_OFFLINE=1

ENV INSPECT_LOG_DIR=/results \
    HOME=/tmp

COPY run_gsm8k_eval.py /opt/gsm8k/run_gsm8k_eval.py

# OpenShift runs as an arbitrary UID in group 0: make the dataset cache and
# results dir group-0 writable with group perms mirroring owner perms.
RUN mkdir -p /results \
    && chgrp -R 0 ${HF_HOME} /results \
    && chmod -R g=u ${HF_HOME} /results

WORKDIR /results

USER 1001

ENTRYPOINT ["python", "/opt/gsm8k/run_gsm8k_eval.py", "--log-dir", "/results"]
CMD []
