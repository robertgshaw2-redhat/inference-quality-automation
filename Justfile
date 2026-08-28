url := env_var_or_default("URL", "http://localhost:8000")
model := env_var_or_default("MODEL", "meta-models/Muse-Glimmer-30B")
bfcl_image := env_var_or_default("BFCL_IMAGE", "bfcl:latest")


bfcl:
	docker run --rm --network host \
	-v "$PWD/bfcl-results:/results:Z" \
	{{bfcl_image}} --model {{model}}