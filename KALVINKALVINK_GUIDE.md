# environment

hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro

# Run

uv run tools/run_webui.py --checkpoint-path checkpoints/s2-pro
uv run tools/run_webui.py --llama-checkpoint-path checkpoints/s2-pro-int4
uv run tools/run_webui.py --llama-checkpoint-path checkpoints/fs-s2-int128-g128-20260519_220525
python tools/api_server.py --llama-checkpoint-path checkpoints/fs-s2-int128-g128-20260519_220525
--decoder-checkpoint-path checkpoints/fs-s2-int128-g128-20260519_220525/codec.pth --listen 0.0.0.0:8080

# quantization

uv run tools/llama/quantize_s2.py --checkpoint-path checkpoints/s2-pro --mode int4

# file structure

| Goal                                | Start here                                                                                      |
|-------------------------------------|-------------------------------------------------------------------------------------------------|
| High-level story (Dual-AR, RVQ, RL) | README.md, docs/en/index.md                                                                     |
| Full pipeline at inference          | fish_speech/inference_engine/__init__.py                                                        |
| Slow + Fast autoregressive model    | fish_speech/models/text2semantic/llama.py (DualARTransformer)                                   |
| Token-by-token generation           | fish_speech/models/text2semantic/inference.py (decode_one_token_ar)                             |
| Audio codec (encode/decode)         | fish_speech/models/dac/modded_dac.py, fish_speech/models/dac/rvq.py                             |
| How text + VQ tokens are packed     | fish_speech/content_sequence.py, fish_speech/datasets/semantic.py                               |
| Training setup                      | fish_speech/models/text2semantic/lit_module.py, fish_speech/configs/text2semantic_finetune.yaml |
| Codec hyperparameters               | fish_speech/configs/modded_dac_vq.yaml                                                          |
| Deeper theory                       | Fish Audio S2 technical report, older Fish-Speech paper                                         |


## Suggested reading order (≈30–60 min)
1. docs/en/index.md — Dual-AR + codec summary (5 min).
2. inference_engine/__init__.py — end-to-end flow (10 min).
3. text2semantic/inference.py — decode_one_token_ar + generate (15 min).
4. text2semantic/llama.py — BaseTransformer, DualARTransformer (20 min).
5. dac/modded_dac.py + modded_dac_vq.yaml — codec side (15 min).
6. ArXiv technical report — training data, RL, benchmarks.





