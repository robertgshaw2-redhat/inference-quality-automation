url := env_var_or_default("URL", "http://localhost:8000")
model := env_var_or_default("MODEL", "meta-models/Muse-Glimmer-30B")
bfcl_image := env_var_or_default("BFCL_IMAGE", "quay.io/rh-ee-robshaw/bfcl:latest")
kvv_image := env_var_or_default("KVV_IMAGE", "quay.io/rh-ee-robshaw/kvv:test")
# The agent runs inside a Docker sandbox whose egress proxy only allows ports
# 80/443, so the server must be reachable from containers on port 80: not
# localhost, and not the usual :8000. See deepswe/README.md.
deepswe_url := env_var_or_default("DEEPSWE_URL", "http://172.17.0.1")
deepswe_tasks := env_var_or_default("DEEPSWE_TASKS", "deepswe/deep-swe/tasks")


bfcl:
	docker run --rm --network host \
	-v "$PWD/bfcl-results:/results:Z" \
	{{bfcl_image}} --model {{model}}

kvv-build:
	docker build -t {{kvv_image}} kvv/

# Run any kvv benchmark in the image; args go straight to vendor/eval.py.
# e.g. just kvv aime2025 --model opensource/{{model}} --max-tokens 98304
kvv *args="":
	docker run --rm --network host \
	-v "$PWD/kvv-results:/results:Z" \
	-e KIMI_BASE_URL={{url}}/v1 \
	-e KIMI_API_KEY=dummy \
	{{kvv_image}} {{args}}

aime-no-thinking:
	docker run --rm --network host \
	-v "$PWD/kvv-results:/results:Z" \
	-e KIMI_BASE_URL={{url}}/v1 \
	-e KIMI_API_KEY=dummy \
	{{kvv_image}} aime2025 \
		--model opensource/{{model}} \
		--think-mode opensource \
		--max-tokens 98304 \
		--stream \
		--epochs 1 \
		--display plain

aime-no-thinking-local:
	export KIMI_BASE_URL={{url}}/v1 && \
	export KIMI_API_KEY=dummy && \
	python3 kvv/vendor/eval.py aime2025 \
		--model opensource/{{model}} \
		--think-mode opensource \
		--max-tokens 98304 \
		--stream \
		--epochs 1 \
		--display plain

# One-time setup: install the Pier runner and clone the DeepSWE tasks.
deepswe-setup:
	uv tool install datacurve-pier
	test -d deepswe/deep-swe || git clone https://github.com/datacurve-ai/deep-swe deepswe/deep-swe

# Run DeepSWE (https://github.com/datacurve-ai/deep-swe) with mini-swe-agent
# against the local OpenAI-compatible server; args go straight to `pier run`.
# e.g. just deepswe --n-tasks 5 --sample-seed 0
#      just deepswe -i abs-stepped-slices
deepswe *args="":
	pier run -p {{deepswe_tasks}} \
		--agent mini-swe-agent \
		--model openai/{{model}} \
		--ae OPENAI_BASE_URL={{deepswe_url}}/v1 \
		--ae OPENAI_API_BASE={{deepswe_url}}/v1 \
		--ae OPENAI_API_KEY=dummy \
		--jobs-dir deepswe-results \
		{{args}}

mmmu-no-thinking-local:
	export KIMI_BASE_URL={{url}}/v1 && \
	export KIMI_API_KEY=dummy && \
	python3 kvv/vendor/eval.py mmmu \
		--model opensource/{{model}} \
		--think-mode opensource \
		--max-tokens 65536 \
		--stream \
		--display plain
