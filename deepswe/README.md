# DeepSWE

Runs [DeepSWE](https://github.com/datacurve-ai/deep-swe) — 113 long-horizon
coding-agent tasks in the Harbor format — against a local OpenAI-compatible
server using [Pier](https://pypi.org/project/datacurve-pier/) and
`mini-swe-agent`.

Unlike the bfcl/kvv images, Pier is not wrapped in our own Dockerfile: it
already orchestrates one Docker sandbox per task (agent container + egress
proxy + separate verifier container), so it runs directly on the host.

## Setup

```bash
just deepswe-setup
```

Installs the `pier` CLI with `uv tool install` and clones the task repo into
`deepswe/deep-swe/` (git-ignored). Requires Docker and Python 3.12+ for uv.

## Running

```bash
# Smoke-test a single task
just deepswe -i abs-stepped-slices

# Deterministic 5-task subset
just deepswe --n-tasks 5 --sample-seed 0

# Full benchmark, 8 tasks at a time
just deepswe -n 8
```

Extra arguments are passed straight to `pier run` (`pier run --help` for the
full list). Results land in `deepswe-results/<job-name>/`; each trial contains
the trajectory plus `verifier/reward.json` with the binary reward. Inspect
trajectories with `pier view`.

## How the model endpoint is wired

DeepSWE tasks declare `network_mode = "no-network"`: the agent container has
no direct internet, only a squid egress proxy restricted to an allowlist.
Pier's `mini-swe-agent` integration forwards `OPENAI_BASE_URL` /
`OPENAI_API_BASE` from `--ae` agent env vars into the sandbox and adds their
host to that allowlist, which is how the recipe points litellm at the local
server.

Two consequences for a locally hosted model:

- `localhost` will not reach a server on the host from inside the sandbox,
  so the recipe defaults to the docker bridge gateway `172.17.0.1`.
- The proxy only permits ports **80 and 443** (squid `Safe_ports`), so the
  usual `:8000` is blocked. Serve on port 80, either directly (vLLM:
  `--host 0.0.0.0 --port 80`, needs root/CAP_NET_BIND_SERVICE) or by
  forwarding: `sudo socat TCP-LISTEN:80,fork,reuseaddr TCP:127.0.0.1:8000`.

```bash
DEEPSWE_URL=http://172.17.0.1 MODEL=meta-models/Muse-Glimmer-30B just deepswe
```

For a remote endpoint on https/443, set `DEEPSWE_URL` to it directly.
