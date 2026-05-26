import time
from pathlib import Path
from typing import Literal

from kui.asgi import HTTPException, request
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from fish_speech.utils.schema import ServeTTSRequest

# Values some browser extensions send when no voice is selected.
_INVALID_VOICE_ALIASES = frozenset({"undefined", "null", "none", ""})

# Built-in OpenAI TTS voice names (no Fish Speech reference; use default voice).
OPENAI_BUILTIN_VOICES = frozenset(
    {
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "onyx",
        "nova",
        "sage",
        "shimmer",
        "verse",
        "default",
    }
)

# OpenAI model ids accepted for speech (actual weights are chosen at server startup).
OPENAI_TTS_MODELS = frozenset({"s2-pro"})

OPENAI_FORMAT_TO_FISH: dict[str, Literal["wav", "pcm", "mp3", "opus"]] = {
    "wav": "wav",
    "pcm": "pcm",
    "mp3": "mp3",
    "opus": "opus",
    "flac": "wav",
}


class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible POST /v1/audio/speech request body."""

    model: str = "tts-1"
    input: str
    voice: str = "default"
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    # Stream WAV chunks as they are synthesized (HTTP chunked). Set false for one-shot file.
    stream: bool = True

    @model_validator(mode="before")
    @classmethod
    def normalize_voice(cls, values):
        if not isinstance(values, dict):
            return values
        voice = values.get("voice")
        if voice is None or (
            isinstance(voice, str) and voice.strip().lower() in _INVALID_VOICE_ALIASES
        ):
            logger.warning(
                "[OpenAI /v1/audio/speech] Invalid or missing voice={!r}; using 'default'",
                voice,
            )
            values["voice"] = "default"
        return values

    def to_serve_tts_request(self) -> ServeTTSRequest:
        if self.model not in OPENAI_TTS_MODELS:
            raise HTTPException(
                400,
                content=f"Unsupported model '{self.model}'. "
                f"Supported: {', '.join(sorted(OPENAI_TTS_MODELS))}",
            )

        if self.response_format == "aac":
            raise HTTPException(
                400,
                content="response_format 'aac' is not supported. Use mp3, opus, wav, or pcm.",
            )

        fish_format = OPENAI_FORMAT_TO_FISH[self.response_format]
        streaming = self.stream
        if streaming and fish_format != "wav":
            logger.warning(
                "[OpenAI /v1/audio/speech] stream=true requires wav chunks; "
                "overriding response_format from {!r} to 'wav'",
                self.response_format,
            )
            fish_format = "wav"

        reference_id = None
        if self.voice not in OPENAI_BUILTIN_VOICES:
            reference_id = self.voice

        if self.speed != 1.0:
            logger.debug(
                "OpenAI 'speed' parameter is not supported by Fish Speech; ignoring."
            )

        return ServeTTSRequest(
            text=self.input,
            format=fish_format,
            reference_id=reference_id,
            streaming=streaming,
        )


def log_openai_speech_request(
    req: OpenAISpeechRequest,
    serve_req: ServeTTSRequest,
    *,
    available_reference_ids: list[str] | None = None,
) -> None:
    """Log parsed OpenAI speech request and Fish Speech mapping (INFO level)."""
    client = getattr(request, "client", None)
    client_addr = f"{client[0]}:{client[1]}" if client else "unknown"
    content_type = request.headers.get("content-type", "")
    auth = request.headers.get("authorization", "")
    auth_hint = "present" if auth else "missing"

    input_preview = req.input[:120] + "..." if len(req.input) > 120 else req.input
    logger.info(
        "[OpenAI /v1/audio/speech] client={} content_type={} authorization={}",
        client_addr,
        content_type,
        auth_hint,
    )
    logger.info(
        "[OpenAI /v1/audio/speech] body: model={!r} voice={!r} response_format={!r} "
        "stream={} speed={} input_len={} input_preview={!r}",
        req.model,
        req.voice,
        req.response_format,
        req.stream,
        req.speed,
        len(req.input),
        input_preview,
    )
    logger.info(
        "[OpenAI /v1/audio/speech] mapped: reference_id={!r} format={!r} "
        "streaming={} builtin_voice={}",
        serve_req.reference_id,
        serve_req.format,
        serve_req.streaming,
        serve_req.reference_id is None,
    )
    if available_reference_ids is not None:
        logger.info(
            "[OpenAI /v1/audio/speech] available reference_ids: {}",
            available_reference_ids or "(none)",
        )
        if (
            serve_req.reference_id
            and serve_req.reference_id not in available_reference_ids
        ):
            logger.warning(
                "[OpenAI /v1/audio/speech] reference_id={!r} was NOT found under "
                "references/ — generation may fail. Use a name from "
                "GET /v1/audio/voices or a built-in voice (alloy, nova, default, ...).",
                serve_req.reference_id,
            )


def log_openai_speech_error(exc: Exception, req: OpenAISpeechRequest | None = None) -> None:
    """Log failures for /v1/audio/speech."""
    if req is not None:
        logger.error(
            "[OpenAI /v1/audio/speech] FAILED with {}: {} | last parsed voice={!r} model={!r}",
            type(exc).__name__,
            exc,
            req.voice,
            req.model,
        )
    else:
        logger.error(
            "[OpenAI /v1/audio/speech] FAILED with {}: {}",
            type(exc).__name__,
            exc,
        )


class OpenAIModel(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "fish-speech"


class OpenAIModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[OpenAIModel]


def build_model_list(checkpoint_path: str) -> OpenAIModelList:
    model_id = Path(checkpoint_path).name or "fish-speech"
    created = int(time.time())
    models = [
        OpenAIModel(id=model_id, created=created),
        OpenAIModel(id="tts-1", created=created),
        OpenAIModel(id="tts-1-hd", created=created),
    ]
    return OpenAIModelList(data=models)


class OpenAIVoice(BaseModel):
    """Voice entry for GET /v1/audio/voices (extensions expect id + name)."""

    id: str
    object: Literal["voice"] = "voice"
    name: str
    created: int
    owned_by: str = "fish-speech"


class OpenAIVoiceList(BaseModel):
    object: Literal["list"] = "list"
    data: list[OpenAIVoice]


def build_voice_list(reference_ids: list[str] | None = None) -> OpenAIVoiceList:
    created = int(time.time())
    voices: list[OpenAIVoice] = []
    for voice_id in sorted(OPENAI_BUILTIN_VOICES):
        voices.append(
            OpenAIVoice(id=voice_id, name=voice_id, created=created),
        )
    for ref_id in sorted(reference_ids or []):
        if ref_id not in OPENAI_BUILTIN_VOICES:
            voices.append(
                OpenAIVoice(id=ref_id, name=ref_id, created=created),
            )
    return OpenAIVoiceList(data=voices)
