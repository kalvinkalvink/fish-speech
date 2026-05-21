# Server

This page covers server-side inference for Fish Audio S2, plus quick links for WebUI inference and Docker deployment.

## API Server Inference

Fish Speech provides an HTTP API server entrypoint at `tools/api_server.py`.

### Start the server locally

```bash
python tools/api_server.py \
  --llama-checkpoint-path checkpoints/s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
  --listen 0.0.0.0:8080
```

Common options:

- `--compile`: enable `torch.compile` optimization
- `--half`: use fp16 mode
- `--api-key`: require bearer token authentication
- `--workers`: set worker process count

CORS is enabled by default for browser and Chrome extension clients (including `chrome-extension://` origins and requests to `http://127.0.0.1`). If you use `--api-key`, set the same key in the extension; preflight `OPTIONS` requests do not require auth.

**Chrome extension still blocked?** Restart the API server after updating, confirm the extension's API URL matches `--listen` (e.g. `http://127.0.0.1:8080`), and add host permission for that host in the extension manifest if required.

### Health check

```bash
curl -X GET http://127.0.0.1:8080/v1/health
```

Expected response:

```json
{"status":"ok"}
```

### Main API endpoints

- `POST /v1/tts` for text-to-speech generation (native Fish Speech API)
- `POST /v1/audio/speech` for OpenAI-compatible text-to-speech
- `GET /v1/models` and `GET /v1/models/{model_id}` for OpenAI-compatible model listing
- `GET /v1/audio/voices` for voice names (use this for extension voice lists, not `/v1/models`)
- `POST /v1/vqgan/encode` for VQ encode
- `POST /v1/vqgan/decode` for VQ decode

### OpenAI-compatible API

The server exposes endpoints compatible with the [OpenAI Audio Speech API](https://platform.openai.com/docs/api-reference/audio/createSpeech), so you can use the official OpenAI Python SDK or any client that calls `/v1/audio/speech`.

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"model":"tts-1","input":"Hello from Fish Speech","voice":"default","response_format":"mp3"}' \
  --output speech.mp3
```

Field mapping:

| OpenAI field | Fish Speech |
|--------------|---------------|
| `input` | `text` |
| `voice` | `reference_id` (use a saved reference id; built-in names like `alloy` use the default voice) |
| `response_format` | `format` (`mp3`, `opus`, `wav`, `pcm`; `flac` maps to `wav`) |
| `model` | Accepted ids: `tts-1`, `tts-1-hd`, `fish-speech` (weights are still loaded at server startup) |

`speed` is accepted for compatibility but not applied. `aac` is not supported.

**Extension voice list textbox:** enter plain voice names, one per line (not JSON). Example:

```
alloy
nova
default
```

Do not paste the response from `GET /v1/models` — that lists **models** (`object: "model"`), not voices. Use `GET /v1/audio/voices` if the extension fetches voices from the API.

Python example with the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",  # omit if --api-key was not set
    base_url="http://127.0.0.1:8080/v1",
)

with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="my-speaker",  # reference_id from /v1/references/list
    input="Hello from Fish Speech",
    response_format="mp3",
) as response:
    response.stream_to_file("speech.mp3")
```

### Python client example

The base TTS model is selected when the server starts. In the example above, the server is started with the `checkpoints/s2-pro` weights, so every request sent to `http://127.0.0.1:8080/v1/tts` will use **S2-Pro** automatically. There is no separate per-request `model` field in `tools/api_client.py` for local server calls.

```bash
python tools/api_client.py \
  --url http://127.0.0.1:8080/v1/tts \
  --text "Hello from Fish Speech" \
  --output s2-pro-demo
```

If you want to select a saved reference voice, use `--reference_id`. This chooses the **voice reference**, not the base TTS model:

```bash
python tools/api_client.py \
  --url http://127.0.0.1:8080/v1/tts \
  --text "Hello from Fish Speech" \
  --reference_id my-speaker \
  --output s2-pro-demo
```

## WebUI Inference

For WebUI usage, see:

- [WebUI Inference](https://speech.fish.audio/inference/#webui-inference)

## Docker

For Docker-based server or WebUI deployment, see:

- [Docker Setup](https://speech.fish.audio/install/#docker-setup)

You can also start the server profile directly with Docker Compose:

```bash
docker compose --profile server up
```
