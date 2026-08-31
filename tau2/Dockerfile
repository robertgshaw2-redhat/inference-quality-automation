FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# tau2-bench is not on PyPI; install from source at a pinned commit (v1.0.1).
# The checkout is kept because the domain task data lives in its data/ dir.
ARG TAU2_REF=a2c024725189473d2d7cea3a5cfdbcc67478e41f
RUN git init /opt/tau2-bench \
    && git -C /opt/tau2-bench remote add origin https://github.com/sierra-research/tau2-bench.git \
    && git -C /opt/tau2-bench fetch --depth 1 origin ${TAU2_REF} \
    && git -C /opt/tau2-bench checkout --detach FETCH_HEAD \
    && uv pip install --system /opt/tau2-bench

ENV TAU2_DATA_DIR=/opt/tau2-bench/data \
    HOME=/tmp

COPY run_tau2_eval.py /opt/tau2/run_tau2_eval.py

# OpenShift runs as an arbitrary UID in group 0:
# - pre-create the /results symlink so the runtime never needs to write into the image
# - make the data dir group-0 writable with group perms mirroring owner perms
RUN mkdir -p /results \
    && ln -s /results ${TAU2_DATA_DIR}/simulations \
    && chgrp -R 0 ${TAU2_DATA_DIR} /results \
    && chmod -R g=u ${TAU2_DATA_DIR} /results

WORKDIR /results

USER 1001

ENTRYPOINT ["python", "/opt/tau2/run_tau2_eval.py", "--output-dir", "/results"]
CMD []