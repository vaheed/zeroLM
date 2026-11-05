# Changelog

## Unreleased

- Added `zerolm serve`, an HTTP/JSON API (std only): an OpenAI-compatible `/v1/chat/completions` and `/v1/models` (models `zerolm`, stateless with the messages as context, and `zerolm-live`, a persistent session per `user`; streaming, `max_tokens`, OpenAI error envelope; tools/functions/`n>1` rejected) plus native `/v1/ask` and `/v1/sessions/{tag}` endpoints. Safe by default: loopback bind, token required elsewhere, request deadline, connection and session caps, shared life-file lock with the CLI. Contract in `rust/openapi.yaml` (validated in CI); docs in `docs/API.md`.
- Added a `zerolm-api` Docker Compose service with a health check (`zerolm health`), 18 API end-to-end tests, and a CI check with the official OpenAI Python SDK.
- Made the `privacy` answer precise about the optional API server (1,053 recurrent states, 515 vocabulary entries now).
- Grew the model from 18 to 28 intents (1,039 recurrent states at the time, 281 features): install, license, performance, languages, version, live, privacy, contribute, topics and comparison. Phrases are tied to ZeroLM ("install zerolm", "how fast is it") rather than bare words, and 13 near-miss prompts ("how do I install Photoshop") were added to the unknown tests. Regression is 119/119 and unknown abstention 31/31.
- Raised `MAX_FEATURES` to 512 and `MAX_VOCAB` to 1024 in the C runtime, Rust port and tools (model structure about 514 KiB).
- Added a std-only Rust port of the runtime (`rust/`) with output parity against the C binary, checked in CI. Bounded parsing replaces fixed C buffers; dangling state references are rejected at load time.
- Added `zerolm live` (ZeroLM Live): persistent decaying memory, rolling transcript for follow-ups, `teach` and good/bad feedback as a learned-reflex overlay, a `--jsonl` daemon mode, and `--tag NAME` sessions with life-file locking so sessions cannot overwrite each other. `weights.zlm` is never modified. See `docs/LIVE.md`.
- Added a Dockerfile and docker-compose.yml (multi-stage, about 117 MB, unprivileged, read-only, no network) with a `docker` CI job and `scripts/docker_smoke.sh`.
- Life files are locked with `flock`, which works across containers sharing a volume and needs no stale-lock handling. A first pid-file version was replaced because each container has its own pid namespace.
- Follow-up resolution now requires a cue word (`why`, `yes`, ...), so gibberish no longer repeats the previous answer.
- Added 16 end-to-end tests of `zerolm live` and `scripts/benchmark_all.py`, which records C vs Rust vs Live speed and memory in `reports/benchmarks.{json,md}`.
- Added `ZEROLM_MODEL` to set the default model path, a clippy CI step, and documentation for the whole project (`INSTALL.md` rewritten, `docs/CI.md` and `docs/LIVE.md` added).

## 0.2.0

- Made the prototype and documentation English-only.
- Expanded from 8 intent paths and 112 recurrent states to 18 intent paths and 694 states.
- Added a JSON model specification and deterministic serializer.
- Optimized the recurrent and output calculations using algebraically equivalent sparse operations.
- Added regression, unknown-input, graph-audit, report, and latency tools.
- Added a patch format and workflow inspired by Weight-Space Programming.
- Added GitHub Actions CI and full architecture/research documentation.

## 0.1.0

- Initial small hand-weighted recurrent text-generation prototype.
