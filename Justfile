url := env_var_or_default("URL", "http://localhost:8000")
# model := env_var_or_default("MODEL", "meta-models/Muse-Glimmer-30B")
model := env_var_or_default("MODEL", "google/gemma-4-12B-it")
bfcl_image := env_var_or_default("BFCL_IMAGE", "quay.io/rh-ee-robshaw/bfcl:latest")
kvv_image := env_var_or_default("KVV_IMAGE", "quay.io/rh-ee-robshaw/kvv:test")
tau2_image := env_var_or_default("TAU2_IMAGE", "quay.io/rh-ee-robshaw/tau2:latest")
gsm8k_image := env_var_or_default("GSM8K_IMAGE", "quay.io/rh-ee-robshaw/gsm8k:latest")

############################################################
# IMAGE BUILDER
############################################################

bfcl-build:
	docker build --ulimit nofile=65536:65536 -t {{bfcl_image}} bfcl/

# BFCL v3 multi-turn via Inspect AI (inspect_evals/bfcl inside the kvv image).
bfcl-no-thinking:
	docker run --rm --network host \
	-v "$PWD/kvv-results:/results:Z" \
	-e KIMI_BASE_URL={{url}}/v1 \
	-e KIMI_API_KEY=dummy \
	{{kvv_image}} bfcl \
		--model opensource/{{model}} \
		--think-mode opensource \
		--max-tokens 4096 \
		--temperature 0.0 \
		--stream \
		--display plain

bfcl-no-thinking-local:
	export KIMI_BASE_URL={{url}}/v1 && \
	export KIMI_API_KEY=dummy && \
	python3 kvv/vendor/eval.py bfcl \
		--model opensource/{{model}} \
		--think-mode opensource \
		--max-tokens 4096 \
		--temperature 0.0 \
		--display plain

kvv-build:
	docker build --ulimit nofile=65536:65536 -t {{kvv_image}} kvv/

tau2-build:
	docker build --ulimit nofile=65536:65536 -t {{tau2_image}} tau2/

gsm8k-build:
	docker build --ulimit nofile=65536:65536 -t {{gsm8k_image}} gsm8k/

############################################################
# TAU2
############################################################

# Run tau2-bench agentic tool calling against the local server; extra args
# go straight to run_tau2_eval.py (and unknown ones on to `tau2 run`).
# e.g. just tau2 --domain airline --num-tasks 5
tau2 *args="":
	docker run --rm --network host --userns=keep-id \
	-v "$PWD/tau2-results:/results:Z" \
	{{tau2_image}} --base-url {{url}}/v1 --model {{model}} {{args}}

# Quick tool-calling smoke test: 5 retail tasks, 1 trial.
tau2-smoke:
	docker run --rm --network host --user $(id -u):$(id -g) --group-add 0 \
	-v "$PWD/tau2-results:/results:Z" \
	{{tau2_image}} --base-url {{url}}/v1 --model {{model}} --domain retail --num-tasks 5 --num-trials 1

# Full retail domain with pass^k reliability over 4 trials.
tau2-retail:
	docker run --rm --network host --user $(id -u):$(id -g) \
	-v "$PWD/tau2-results:/results:Z" \
	{{tau2_image}} --base-url {{url}}/v1 --model {{model}} --domain retail --num-trials 4

############################################################
# GSM8K
############################################################

# Run GSM8K (grade-school math, Inspect AI harness) against the local
# server; extra args go straight to run_gsm8k_eval.py.
gsm8k *args="":
	mkdir -p gsm8k-results
	docker run --rm --network host --user $(id -u):$(id -g) --group-add 0 \
	-v "$PWD/gsm8k-results:/results:Z" \
	{{gsm8k_image}} --base-url {{url}}/v1 --model {{model}} {{args}}

############################################################
# AIME2025
############################################################

aime-thinking epochs="1":
	docker run --rm --network host \
		-e KIMI_BASE_URL="{{url}}/v1" \
		-e KIMI_API_KEY="dummy" \
		-v "$PWD/aime-results:/results:Z" \
		{{kvv_image}} aime2025 \
			--model opensource/{{model}} \
			--think-mode opensource \
			--thinking \
			--max-tokens 98304 \
			--stream \
			--epochs {{epochs}} \
			--display plain

aime-no-thinking epochs="1":
	docker run --rm --network host \
	-e KIMI_BASE_URL="{{url}}/v1" \
	-e KIMI_API_KEY="dummy" \
	-v "$PWD/aime-results:/results:Z" \
	{{kvv_image}} aime2025 \
		--model opensource/{{model}} \
		--think-mode opensource \
		--max-tokens 98304 \
		--stream \
		--epochs {{epochs}} \
		--display plain


############################################################
# MMMU
############################################################

mmmu-no-thinking:
	export KIMI_BASE_URL={{url}}/v1 && \
	export KIMI_API_KEY=dummy && \
	python3 kvv/vendor/eval.py mmmu \
		--model opensource/{{model}} \
		--think-mode opensource \
		--max-tokens 65536 \
		--stream \
		--display plain
