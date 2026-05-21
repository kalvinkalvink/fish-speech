# environment
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
# Run
uv run tools/run_webui.py --checkpoint-path checkpoints/s2-pro
uv run tools/run_webui.py --llama-checkpoint-path checkpoints/s2-pro-int4
uv run tools/run_webui.py --llama-checkpoint-path checkpoints/fs-s2-int128-g128-20260519_220525
python tools/api_server.py --llama-checkpoint-path checkpoints/fs-s2-int128-g128-20260519_220525 --decoder-checkpoint-path checkpoints/fs-s2-int128-g128-20260519_220525/codec.pth --listen 0.0.0.0:8080
# quantization
uv run tools/llama/quantize_s2.py --checkpoint-path checkpoints/s2-pro --mode int4
