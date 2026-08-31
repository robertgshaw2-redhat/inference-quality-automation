url := env_var_or_default("URL", "http://localhost:8000")
# model := env_var_or_default("MODEL", "meta-models/Muse-Glimmer-30B")
model := env_var_or_default("MODEL", "google/gemma-4-12B-it")
bfcl_image := env_var_or_default("BFCL_IMAGE", "quay.io/rh-ee-robshaw/bfcl:latest")
kvv_image := env_var_or_default("KVV_IMAGE", "quay.io/rh-ee-robshaw/kvv:test")
tau2_image := env_var_or_default("TAU2_IMAGE", "quay.io/rh-ee-robshaw/tau2:latest")
gsm8k_image := env_var_or_default("GSM8K_IMAGE", "quay.io/rh-ee-robshaw/gsm8k:latest")
terminalbench_image := env_var_or_default("TERMINALBENCH_IMAGE", "quay.io/rh-ee-robshaw/terminalbench:latest")

############################################################
# IMAGE BUILDER
############################################################

bfcl-build:
	docker build --ulimit nofile=65536:65536 -t {{bfcl_image}} bfcl/

kvv-build:
	docker build --ulimit nofile=65536:65536 -t {{kvv_image}} kvv/

tau2-build:
	docker build --ulimit nofile=65536:65536 -t {{tau2_image}} tau2/

gsm8k-build:
	docker build --ulimit nofile=65536:65536 -t {{gsm8k_image}} gsm8k/

terminalbench-build:
	docker build --ulimit nofile=65536:65536 -t {{terminalbench_image}} terminalbench/

fix-perms:
    docker run --rm -v "$PWD/tau2-results:/results" --entrypoint /bin/sh {{tau2_image}} -c "chown -R $(id -u):$(id -g) /results"

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
# e.g. just gsm8k --fewshot 0 --epochs 4
gsm8k *args="":
	docker run --rm --network host --user $(id -u):$(id -g) \
	-v "$PWD/gsm8k-results:/results:Z" \
	{{gsm8k_image}} --base-url {{url}}/v1 --model {{model}} {{args}}

# Quick smoke test: 10 problems.
gsm8k-smoke:
	docker run --rm --network host --user $(id -u):$(id -g) \
	-v "$PWD/gsm8k-results:/results:Z" \
	{{gsm8k_image}} --base-url {{url}}/v1 --model {{model}} --num-problems 10

############################################################
# TERMINAL-BENCH 2.1
############################################################

# Run Terminal-Bench 2.1 (89 terminal tasks, Harbor harness) against the
# local server; extra args go straight to run_terminalbench_eval.py (and
# unknown ones on to `harbor run`). Harbor launches one container per task
# on the HOST docker daemon through the mounted socket, which is why the
# results dir is mounted at the same absolute path on both sides: harbor
# bind-mounts trial dirs from it into the task containers, and those mounts
# resolve on the host. e.g. just terminalbench --task hello-world
terminalbench *args="":
	mkdir -p terminalbench-results
	docker run --rm --network host \
	--user $(id -u):$(id -g) --group-add $(stat -c '%g' /var/run/docker.sock) \
	-v /var/run/docker.sock:/var/run/docker.sock \
	-v "$PWD/terminalbench-results:$PWD/terminalbench-results:z" \
	{{terminalbench_image}} --base-url {{url}}/v1 --model {{model}} \
	--jobs-dir "$PWD/terminalbench-results" {{args}}

# Quick smoke test: 3 tasks, 2 at a time.
terminalbench-smoke:
	just terminalbench --n-tasks 3 --n-concurrent 2

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
