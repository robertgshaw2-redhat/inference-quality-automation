# KVV — Kimi Vendor Verifier eval image

Containerized [MoonshotAI/Kimi-Vendor-Verifier](https://github.com/MoonshotAI/kimi-vendor-verifier)
for running inference-quality evaluations — with a focus on tool calling and
API-contract correctness — against an already-running OpenAI-compatible
endpoint, such as an [llm-d](https://llm-d.ai) deployment.

## What's in here

| Path | Purpose |
|------|---------|
| `vendor/` | Vendored upstream source (see [Provenance](#provenance)) |
| `run_kvv_eval.py` | Entrypoint: runs selected suites against a server, writes reports |
| `Dockerfile` | Builds the eval image from the vendored source + locked deps |
| `k8s/kvv-job.yaml` | Kubernetes Job template for running evals on a cluster |

## Suites

Selected with `--suites` (comma- or space-separated, or `all`):

| Suite | Type | What it checks |
|-------|------|----------------|
| `params` | pytest | API parameter constraints (`temperature`, `top_p`, `n`, …) |
| `tool_schema` | pytest | Tool-call `function.arguments` validate against walle JSON schemas, streaming and non-streaming |
| `k3_features` | pytest | K3 feature contract: dynamic tools, `response_format`, `tool_choice`, thinking effort |
| `prompt_tokens` | pytest | Vendor-reported `usage.prompt_tokens` accuracy |
| `ocrbench` | inspect-ai | OCR text recognition (vision models) |
| `mmmu` | inspect-ai | MMMU Pro Vision (vision models) |
| `aime2025` | inspect-ai | AIME 2025 math reasoning (32 epochs by default) |

The default is the four pytest verifier suites — the tool-calling and
API-contract checks. The inspect-ai benchmarks download their datasets from
Hugging Face on first run, so the pod needs egress for those.

Note: the verifier suites encode the Kimi K3 API contract. Run them against
Kimi models served by vLLM/SGLang-based stacks (llm-d) with the default
`--think-mode opensource`; some `k3_features` tests are expected to fail on
non-Kimi models.

## Build

```bash
docker build -t kvv:latest kvv/
# or: just kvv-build
```

## Run locally against a server on localhost:8000

```bash
docker run --rm --network host -v "$PWD/kvv-results:/results:Z" kvv:latest
# or: just kvv
```

Model is auto-detected from `/v1/models`; pass `--model` to override. All
runner flags:

```bash
docker run --rm kvv:latest --help
```

Common variations:

```bash
# Tool-call schema suite only, thinking on, capped at 50 cases
just kvv --suites tool_schema --thinking --max-cases 50

# Remote endpoint
URL=http://my-llm-d-gateway:8000 just kvv

# Vision benchmark sanity check
just kvv --suites ocrbench --thinking --thinking-effort high
```

## Run on a cluster against llm-d

1. Push the image somewhere the cluster can pull:

   ```bash
   docker build -t <registry>/kvv:latest kvv/
   docker push <registry>/kvv:latest
   ```

2. Edit `k8s/kvv-job.yaml`: set `image`, point `KIMI_BASE_URL` at your llm-d
   inference gateway service (including `/v1`), and pick suites/flags in
   `args`.

3. Apply and watch:

   ```bash
   kubectl apply -f kvv/k8s/kvv-job.yaml
   kubectl logs -f job/kvv-eval
   ```

## Outputs

Written to `/results` (mount a volume to keep them):

| File | Contents |
|------|----------|
| `summary.json` | Per-suite pass/fail and exit codes; overall `passed` |
| `junit-<suite>.xml` | JUnit XML per pytest suite |
| `tool-call-schema-report.json` | Per-case outcomes + summary for `tool_schema` |
| `logs/` | inspect-ai `.eval` logs for benchmark suites (view with `inspect view`) |

The container exits 0 only if every requested suite passed, so a Kubernetes
Job's status reflects the eval result.

## Provenance

`vendor/` is a verbatim copy of
[MoonshotAI/kimi-vendor-verifier](https://github.com/MoonshotAI/kimi-vendor-verifier)
at commit `3dad65a760a8867cda72f6dd8848d876a4e851b4` (2026-08-13), MIT
licensed (`vendor/LICENSE`), with two mechanical changes:

- Git LFS pointer files (`beam/data/*`, `testdata/prompt_token_cases/*`) were
  replaced with their real contents, sha256-verified against the pointer oids,
  so the repo and image are self-contained without git-lfs.
- `.gitattributes` (LFS filter config) was dropped accordingly.

To update: re-clone upstream, re-materialize LFS files, replace `vendor/`,
and update the commit hash here.
