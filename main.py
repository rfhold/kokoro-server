import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator, Iterator
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from audio_writer import CONTENT_TYPES, StreamingAudioWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = os.getenv("KOKORO_DEVICE", "cuda")
REPO_ID = os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")
EXECUTOR_WORKERS = int(os.getenv("EXECUTOR_WORKERS", "1"))
SAMPLE_RATE = 24000

LANG_CODES = ["a", "b", "e", "f", "h", "i", "j", "p", "z"]
VOICE_PREFIX_TO_LANG = {lang: lang for lang in LANG_CODES}

pipelines: dict[str, object] = {}
executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)


def _configure_otel(app: FastAPI) -> None:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "kokoro-service")}
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="/health$")


def _resolve_lang_code(voice: str, lang_code: str | None) -> str:
    if lang_code in LANG_CODES:
        return lang_code
    prefix = voice[0] if voice else ""
    return VOICE_PREFIX_TO_LANG.get(prefix, "a")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    from kokoro import KPipeline
    from kokoro.model import KModel

    logger.info("Loading Kokoro model on %s from %s", DEVICE, REPO_ID)
    model = KModel(repo_id=REPO_ID).to(DEVICE).eval()
    for lang in LANG_CODES:
        try:
            pipelines[lang] = KPipeline(lang_code=lang, repo_id=REPO_ID, model=model)
            logger.info("Loaded pipeline for lang_code='%s'", lang)
        except Exception as exc:
            logger.warning("Failed to load pipeline for lang_code='%s': %s", lang, exc)
    logger.info("Loaded %d/%d pipelines", len(pipelines), len(LANG_CODES))
    yield
    pipelines.clear()
    executor.shutdown(wait=False)


app = FastAPI(
    title="Kokoro TTS Service",
    description="OpenAI-compatible text-to-speech",
    version="1.0.0",
    lifespan=lifespan,
)
_configure_otel(app)


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "af_heart"
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    lang_code: str | None = None


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, _: Exception) -> JSONResponse:
    logger.exception("Unhandled exception for %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "healthy" if pipelines else "loading",
            "device": DEVICE,
            "model": REPO_ID,
            "languages": list(pipelines.keys()),
        }
    )


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "kokoro-tts", "docs": "/docs"}


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest) -> StreamingResponse:
    if len(request.input) > 4096:
        raise HTTPException(
            status_code=400,
            detail="Input text exceeds maximum length of 4096 characters",
        )
    if not request.input:
        raise HTTPException(status_code=400, detail="Input text must not be empty")

    lang = _resolve_lang_code(request.voice, request.lang_code)
    pipeline = pipelines.get(lang)
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable for the requested voice",
        )

    def generate_audio() -> Iterator[bytes]:
        writer = StreamingAudioWriter(request.response_format, SAMPLE_RATE)
        try:
            for result in pipeline(
                request.input, voice=request.voice, speed=request.speed
            ):
                if result.audio is not None:
                    chunk = writer.write_chunk(result.audio.numpy())
                    if chunk:
                        yield chunk
            final = writer.finalize()
            if final:
                yield final
        finally:
            writer.close()

    async def stream() -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        generator = generate_audio()

        def get_next_chunk() -> bytes | None:
            try:
                return next(generator)
            except StopIteration:
                return None

        while True:
            chunk = await loop.run_in_executor(executor, get_next_chunk)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(
        stream(),
        media_type=CONTENT_TYPES[request.response_format],
        headers={
            "Content-Disposition": (
                f"attachment; filename=speech.{request.response_format}"
            ),
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )


@app.get("/v1/audio/voices")
async def list_voices() -> dict[str, list[dict[str, str]]]:
    try:
        from huggingface_hub import list_repo_tree

        entries = list_repo_tree(REPO_ID, path_in_repo="voices")
        voices = [
            {
                "id": entry.path.split("/")[-1].removesuffix(".pt"),
                "name": entry.path.split("/")[-1].removesuffix(".pt"),
            }
            for entry in entries
            if entry.path.endswith(".pt")
        ]
    except Exception:
        voices = [{"id": "af_heart", "name": "af_heart"}]
    return {"voices": voices}


@app.get("/v1/models")
async def list_models() -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1686935002,
                "owned_by": "kokoro",
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1686935002,
                "owned_by": "kokoro",
            },
        ],
    }
