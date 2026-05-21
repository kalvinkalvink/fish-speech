import traceback
from http import HTTPStatus

from kui.asgi import HTTPException, JSONResponse, request
from loguru import logger

from tools.server.cors_utils import apply_cors_to_response


def _is_openai_speech_path() -> bool:
    path = getattr(request, "path", "") or ""
    return "/audio/speech" in path


class ExceptionHandler:

    async def http_exception_handler(self, exc: HTTPException):
        if _is_openai_speech_path():
            logger.warning(
                "[OpenAI /v1/audio/speech] HTTP {} {} — message={!r}",
                request.method,
                request.path,
                exc.content,
            )
        return apply_cors_to_response(
            JSONResponse(
                dict(
                    statusCode=exc.status_code,
                    message=exc.content,
                    error=HTTPStatus(exc.status_code).phrase,
                ),
                exc.status_code,
                exc.headers,
            )
        )

    async def other_exception_handler(self, exc: Exception):
        if _is_openai_speech_path():
            logger.error(
                "[OpenAI /v1/audio/speech] {} {} — {}: {}",
                request.method,
                request.path,
                type(exc).__name__,
                exc,
            )
        traceback.print_exc()

        status = HTTPStatus.INTERNAL_SERVER_ERROR
        return apply_cors_to_response(
            JSONResponse(
                dict(statusCode=status, message=str(exc), error=status.phrase),
                status,
            )
        )
