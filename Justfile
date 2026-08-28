url := env_var_or_default("URL", "http://localhost:8000")
model := env_var_or_default("MODEL", "meta-models/Muse-Glimmer-30B")
bfcl_image := env_var_or_default("BFCL_IMAGE", "quay.io/rh-ee-robshaw/bfcl:latest")
kvv_image := env_var_or_default("KVV_IMAGE", "quay.io/rh-ee-robshaw/kvv:test")
tau2_image := env_var_or_default("TAU2_IMAGE", "quay.io/rh-ee-robshaw/tau2:latest")


bfcl:
	docker run --rm --network host \
	-v "$PWD/bfcl-results:/results:Z" \
	{{bfcl_image}} --model {{model}}

kvv-build:
	docker build -t {{kvv_image}} kvv/

tau2-build:
	docker build -t {{tau2_image}} tau2/

# Run tau2-bench agentic tool calling against the local server; extra args
# go straight to run_tau2_eval.py (and unknown ones on to `tau2 run`).
# e.g. just tau2 --domain airline --num-tasks 5
tau2 *args="":
	docker run --rm --network host \
	-v "$PWD/tau2-results:/results:Z" \
	{{tau2_image}} --model {{model}} {{args}}

# Quick tool-calling smoke test: 5 retail tasks, 1 trial.
tau2-smoke:
	docker run --rm --network host \
	-v "$PWD/tau2-results:/results:Z" \
	{{tau2_image}} --model {{model}} --domain retail --num-tasks 5

# Full retail domain with pass^k reliability over 4 trials.
tau2-retail:
	docker run --rm --network host \
	-v "$PWD/tau2-results:/results:Z" \
	{{tau2_image}} --model {{model}} --domain retail --num-trials 4

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

mmmu-no-thinking-local:
	export KIMI_BASE_URL={{url}}/v1 && \
	export KIMI_API_KEY=dummy && \
	python3 kvv/vendor/eval.py mmmu \
		--model opensource/{{model}} \
		--think-mode opensource \
		--max-tokens 65536 \
		--stream \
		--display plain
