# Kokoro Server

Kokoro Server provides an OpenAI-compatible text-to-speech API backed by
`hexgrad/Kokoro-82M`. The production image targets NVIDIA CUDA on AMD64 and
runs one Uvicorn worker.

## API

`POST /v1/audio/speech` accepts the OpenAI speech fields `model`, `input`,
`voice`, `response_format`, and `speed`, plus the optional `lang_code` field.
Supported response formats are `mp3`, `opus`, `aac`, `flac`, `wav`, and `pcm`.

Production exposes the endpoint at both locations:

- `https://kokoro.holdenitdown.net/v1/audio/speech`
- `https://agent-gateway.holdenitdown.net/v1/audio/speech`

The legacy hostname routes all paths. The Agent Gateway hostname routes only
an exact `POST /v1/audio/speech` match.

## Configuration

| Variable                      | Purpose                          | Default              |
| ----------------------------- | -------------------------------- | -------------------- |
| `KOKORO_DEVICE`               | PyTorch inference device         | `cuda`               |
| `KOKORO_REPO_ID`              | Hugging Face model repository    | `hexgrad/Kokoro-82M` |
| `EXECUTOR_WORKERS`            | Audio synthesis executor threads | `1`                  |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Enables OTLP tracing when set    | unset                |
| `OTEL_SERVICE_NAME`           | OpenTelemetry service name       | `kokoro-service`     |

The first model load downloads model data to the Hugging Face cache. The
deployment mounts the 5 Gi `shared-fs` PVC `kokoro-model-cache` at that cache
path and runs one replica.

## Development

Create an environment and run the API tests:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/pytest
```

Run syntax and manifest checks:

```bash
python3 -m compileall -q main.py audio_writer.py tests
kustomize build manifests >/dev/null
```

The API tests replace the loaded synthesis pipeline with an in-memory fake, so
they do not require a GPU or download model data.

The guarded ownership cutover and rollback procedure is maintained in the
homelab repository at `docs/operations/speech-service-extraction.md`.

## Repository Map

| Path                    | Responsibility                                             |
| ----------------------- | ---------------------------------------------------------- |
| `main.py`               | FastAPI application, model lifecycle, and speech endpoints |
| `audio_writer.py`       | Streaming audio encoding                                   |
| `Dockerfile.cuda-amd64` | CUDA AMD64 production image                                |
| `tests/`                | API behavior tests without model loading                   |
| `manifests/`            | `kokoro` namespace Kubernetes resources and routes         |
| `.tekton/push.yaml`     | AMD64 BuildKit image build and Pantheon deployment         |
