# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-10-11

### Added

- `POST /v1/systemone` — the typed decision endpoint at the industry-standard System One path

### Deprecated

- `POST /v1/decisions` — still served as an alias of `/v1/systemone`; migrate callers

## [0.2.0] - 2026-09-22

### Fixed

- A model worker that exits after becoming READY (crash, OOM kill, external kill) is now reaped on the next request, status read or watcher tick, and the model reloads; previously the manager kept reporting READY with the stale pid and every proxied request failed with 500.


### Added

- New `decision` model type for Laya typed decision models (choice/score/noul)
- `POST /v1/decisions` endpoint — takes `state` + typed `questions`, returns probabilities
- `laya-en` and `laya-multilingual` pre-configured in `models.yaml`
- In-process lifecycle for decision models (same as embeddings/TTS/STT)
- `laya-mlx>=0.1.0` optional dependency group (`pip install mlx-serve[decision]`)

## [0.1.0] - 2025-03-31

### Added

- OpenAI-compatible API endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/speech`, `/v1/audio/transcriptions`)
- Single-model memory enforcement with automatic hot-swapping
- Support for text, vision, embedding, TTS, and STT model types
- Subprocess isolation for text/vision models, in-process for embeddings
- YAML-based model configuration with multi-path discovery
- CLI with `init`, `start`, `stop`, `status`, and `logs` commands
- Observability: request metrics, memory monitoring, event logging
- JSONL audit logs for requests and model lifecycle events
- Dashboard endpoint (`/v1/dashboard`) for at-a-glance status
