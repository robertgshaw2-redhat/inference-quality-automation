url := env_var_or_default("URL", "http://localhost:8000")
model := env_var_or_default("MODEL", "meta-models/Muse-Glimmer-30B")
bfcl_image := env_var_or_default("BFCL_IMAGE", "quay.io/rh-ee-robshaw/bfcl:latest")
kvv_image := env_var_or_default("KVV_IMAGE", "quay.io/rh-ee-robshaw/kvv:test")


bfcl:
	docker run --rm --network host \
	-v "$PWD/bfcl-results:/results:Z" \
	{{bfcl_image}} --model {{model}}

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

mmmu-no-thinking-local:
	export KIMI_BASE_URL={{url}}/v1 && \
	export KIMI_API_KEY=dummy && \
	python3 kvv/vendor/eval.py mmmu \
		--model opensource/{{model}} \
		--think-mode opensource \
		--max-tokens 65536 \
		--stream \
		--display plain
