url := env_var_or_default("URL", "http://localhost:8000")
model := env_var_or_default("MODEL", "meta-models/Muse-Glimmer-30B")
bfcl_image := env_var_or_default("BFCL_IMAGE", "bfcl:latest")
kvv_image := env_var_or_default("KVV_IMAGE", "kvv:latest")


bfcl:
	docker run --rm --network host \
	-v "$PWD/bfcl-results:/results:Z" \
	{{bfcl_image}} --model {{model}}

kvv-build:
	docker build -t {{kvv_image}} kvv/

# Model is auto-detected from the server unless MODEL_NAME/--model is given.
kvv *args="":
	docker run --rm --network host \
	-v "$PWD/kvv-results:/results:Z" \
	{{kvv_image}} --base-url {{url}}/v1 {{args}}