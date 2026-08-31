FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Harbor drives the per-task containers through the *host's* Docker daemon
# via the mounted /var/run/docker.sock, so only the docker CLI and the
# compose plugin are needed in this image, not a daemon.
ARG DOCKER_VERSION=28.3.2
ARG COMPOSE_VERSION=v2.39.2
RUN arch="$(uname -m)" \
    && curl -fsSL "https://download.docker.com/linux/static/stable/${arch}/docker-${DOCKER_VERSION}.tgz" \
        | tar -xz --strip-components=1 -C /usr/local/bin docker/docker \
    && mkdir -p /usr/local/lib/docker/cli-plugins \
    && curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-compose \
        "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-${arch}" \
    && chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# Harbor (https://harborframework.com) is the official Terminal-Bench 2.x
# harness; its CLI surface still moves between minor versions, so pin exact.
RUN uv pip install --system "harbor==0.22.0"

# Bake the Terminal-Bench 2.1 dataset (89 tasks) into the image so runs
# don't need Harbor registry access; the wrapper runs `harbor run -p` against
# this path unless --dataset is passed.
RUN harbor download terminal-bench/terminal-bench-2-1 -o /opt/terminal-bench
ENV TB_DATASET_PATH=/opt/terminal-bench/terminal-bench-2-1

# Harbor writes caches under $HOME; /tmp stays writable for any UID.
ENV HOME=/tmp

COPY run_terminalbench_eval.py /opt/terminalbench/run_terminalbench_eval.py

# OpenShift-style arbitrary UID in group 0: make the dataset and results dir
# group-0 writable with group perms mirroring owner perms.
RUN mkdir -p /results \
    && chgrp -R 0 /opt/terminal-bench /results \
    && chmod -R g=u /opt/terminal-bench /results

WORKDIR /results

USER 1001

ENTRYPOINT ["python", "/opt/terminalbench/run_terminalbench_eval.py"]
CMD []
