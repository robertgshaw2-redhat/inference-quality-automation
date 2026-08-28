# TCV — Tool-Call Verifier eval image

Model-agnostic tool-calling quality evals for OpenAI-compatible endpoints,
in the spirit of the
[Kimi-Vendor-Verifier tests](https://github.com/MoonshotAI/Kimi-Vendor-Verifier/tree/main/tests)
but built to cover other model families — GLM first — behind per-family
profiles, with a curated case baseline that's designed to grow over time.

Runs against an *already-running* server, such as an
[llm-d](https://llm-d.ai) deployment; it never starts one.

## What's in here

| Path | Purpose |
|------|---------|
| `cases/` | Curated behavior cases (JSONL) — **the growing baseline** |
| `tcv/` | Harness library: profiles, case engine, streaming reassembly |
| `tests/behavior/` | Runs every case in `cases/`, non-streaming and streaming |
| `tests/tool_choice/` | tool_choice contract: auto / none / required / named |
| `tests/schema/` | Tool-call arguments fuzzed against walle JSON Schemas |
| `testdata/` | walle schema corpus (see [Provenance](#provenance)) |
| `run_tcv_eval.py` | Entrypoint: runs selected suites, writes reports |
| `Dockerfile` | Builds the eval image |
| `k8s/tcv-job.yaml` | Kubernetes Job template |

## Suites

| Suite | What it checks |
|-------|----------------|
| `behavior` | Each curated case: right tool called, arguments are valid JSON matching the tool schema, value-level checks (types, escaping, unicode), parallel calls, multi-turn tool-result round trips. Every case runs in **both** non-streaming and streaming mode — streaming reassembly is where serving-stack tool-call parsers most often break. |
| `tool_choice` | `auto` follows the prompt, `none` forbids calls, `required` forces one, named-function form forces that exact tool (profile-gated). |
| `schema` | Every walle-valid JSON Schema becomes a tool's `parameters` with `tool_choice="required"`; returned `function.arguments` must validate against it (jsonschema Draft 2020-12), streaming and non-streaming. |

## Model-family profiles

The suites are model-agnostic; the per-family differences live in
`tcv/profiles.py`:

| Profile | Thinking toggle (`chat_template_kwargs`) | Notes |
|---------|------------------------------------------|-------|
| `glm` (default) | `enable_thinking` | GLM-4.5/4.6 on vLLM/SGLang |
| `qwen3` | `enable_thinking` | |
| `kimi` | `thinking` | named tool_choice unsupported |
| `deepseek` | `thinking` | V3.1+ hybrid thinking |
| `generic` | none sent | unknown families |

`--thinking default` (the default) sends nothing at all, so the server's
template default applies; `on`/`off` send the profile's toggle explicitly.

For GLM on vLLM, serve with the matching parsers so tool calls come back
structured:

```bash
vllm serve zai-org/GLM-4.6 \
  --tool-call-parser glm45 --reasoning-parser glm45 \
  --enable-auto-tool-choice
```

## Build and run

```bash
docker build -t tcv:latest tcv/
# or: just tcv-build

# Against a server on localhost:8000 (model auto-detected from /v1/models)
docker run --rm --network host -v "$PWD/tcv-results:/results:Z" tcv:latest
# or: just tcv

# Common variations
just tcv --suites behavior --thinking off
just tcv --suites schema --max-cases 50
just tcv --profile qwen3 --tags parsing,parallel
URL=http://my-llm-d-gateway:8000 just tcv
```

Uncontainerized, from this directory:

```bash
uv sync
uv run python run_tcv_eval.py --base-url http://localhost:8000/v1
```

For cluster runs, edit and apply `k8s/tcv-job.yaml` (same workflow as
`kvv/k8s/kvv-job.yaml`).

## Outputs

Written to `/results` (mount a volume to keep them):

| File | Contents |
|------|----------|
| `summary.json` | Per-suite pass/fail and exit codes; overall `passed` |
| `junit-<suite>.xml` | JUnit XML per suite |
| `behavior-report.json` | Per-case × per-mode outcomes + summary |
| `schema-report.json` | Per-schema-case outcomes + summary |

The container exits 0 only if every requested suite passed.

## Growing the baseline: adding cases

This is the intended workflow — when you hit a tool-call parsing bug in the
wild, distill it into a case so it stays covered forever:

1. Append a JSON line to an existing `cases/*.jsonl` file (or add a new
   file for a new category — every `*.jsonl` in `cases/` is picked up
   automatically).
2. Give it a unique `id`, a `description`, and `tags`.
3. Re-run `just tcv --suites behavior --case-filter <id>` to try just that
   case.

Case format (see `tcv/cases.py` for the full reference):

```json
{
  "id": "my_new_case",
  "description": "what regression this pins down",
  "tags": ["parsing"],
  "request": {
    "messages": [{"role": "user", "content": "..."}],
    "tools": [{"type": "function", "function": {"name": "...", "parameters": {...}}}],
    "tool_choice": "auto"
  },
  "expect": {
    "tool_calls": {
      "min": 1, "max": 1,
      "names": ["my_tool"],
      "checks": [{"path": "some.field", "contains_any": ["expected"]}]
    }
  }
}
```

Expectation vocabulary:

- `tool_calls`: `min`/`max` call counts, `names` (allowed set),
  `require_names` (each must appear), `args_match_schema` (default true —
  arguments validate against the tool's declared `parameters`), and
  `checks` — value assertions with a dot `path` into the parsed arguments
  (`items.0.sku`), one op each (`equals`, `one_of`, `contains`,
  `contains_any`, `json_type`), and `call` (index or `"any"`).
- `no_tool_calls: true` + `content` (`nonempty`, `contains`,
  `contains_any`) for text-answer cases.
- `finish_reason` to override the default expectation (`"tool_calls"` when
  calls are expected, `"stop"` otherwise; `"any"` disables the check).
- Arguments must always parse as a JSON object — an empty or truncated
  `arguments` string fails regardless of checks.

Guidelines for good cases:

- Deterministic on a well-behaved model: assert what *must* be true
  (structure, types, verbatim substrings the prompt demands), not exact
  phrasings the model chooses.
- One failure mode per case, named by the `id`.
- Prefer `contains_any` with a few spellings over `equals` for
  free-text-derived values.

To add a new model family, add a `Profile` entry in `tcv/profiles.py`.

## Provenance

`testdata/walle_validator_cases/` and the schema-wrapping logic in
`tcv/schema_cases.py` are adapted from
[MoonshotAI/kimi-vendor-verifier](https://github.com/MoonshotAI/kimi-vendor-verifier)
(MIT), via the vendored copy in `../kvv/vendor` at the commit pinned in
`kvv/README.md`. The behavior suite, profiles, and runner are first-party.
