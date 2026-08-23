# Index

| Path                                     | Info                                                                 |
| ---------------------------------------- | -------------------------------------------------------------------- |
| [`README.md`](README.md)                 | API contract, configuration, validation, and repository orientation. |
| [`main.py`](main.py)                     | OpenAI-compatible speech API and Kokoro model lifecycle.             |
| [`audio_writer.py`](audio_writer.py)     | Streaming audio format encoding.                                     |
| [`tests/`](tests/)                       | API tests that must not load models or require a GPU.                |
| [`manifests/`](manifests/)               | Kubernetes resources for the `kokoro` namespace.                     |
| [`.tekton/push.yaml`](.tekton/push.yaml) | BuildKit build and Pantheon deployment pipeline.                     |

# Hints

- Preserve `POST /v1/audio/speech` compatibility and `hexgrad/Kokoro-82M` unless a behavior change explicitly requires otherwise.
- Keep production on CUDA AMD64 with one Uvicorn worker and one synthesis executor.
- Keep tests independent of GPUs, model downloads, and live infrastructure.
- Validate Python compilation, API tests, YAML parsing, and `kustomize build manifests` after changes.
- Do not commit, publish images, dispatch pipelines, deploy, or query live infrastructure without explicit authorization.
