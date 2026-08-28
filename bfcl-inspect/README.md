# BFCL multi-turn via Inspect AI

Runs the BFCL v3 multi-turn tool-calling categories against an already running
OpenAI-compatible server, using the [`inspect_evals/bfcl`](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/bfcl)
task on the [Inspect AI](https://inspect.aisi.org.uk/) framework instead of
BFCL's own harness (see `../bfcl/` for that).

Default categories are BFCL's `multi_turn` group: `multi_turn_base`,
`multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context`
(200 stateful conversations each; scored by state-based + response-based
checking, so a sample only passes if every turn is correct).

## Usage

```bash
# Build the image
just bfcl-inspect-build

# Full multi-turn run against $URL (default http://localhost:8000)
just bfcl-inspect

# Smoke test: first 5 conversations per category
just bfcl-inspect --num-prompts 5

# Specific categories
just bfcl-inspect --test-category multi_turn_base,multi_turn_miss_func

# Without docker (needs: uv pip install "inspect-ai>=0.3.258" "inspect-evals[bfcl]" openai)
just bfcl-inspect-local --num-prompts 5
```

Outputs land in `./bfcl-inspect-results/`:

- `bfcl_inspect_summary.json` — per-category accuracy summary
- `logs/*.eval` — full Inspect transcripts; browse with `inspect view --log-dir bfcl-inspect-results/logs`
