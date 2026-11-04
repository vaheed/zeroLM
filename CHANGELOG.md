# Changelog

## Unreleased

- Grew the model from 18 to 28 intents (1,039 recurrent states, 281 features): install, license, performance, languages, version, live, privacy, contribute, topics and comparison. Phrases are tied to ZeroLM ("install zerolm", "how fast is it") rather than bare words, and 13 near-miss prompts ("how do I install Photoshop") were added to the unknown tests. Regression is 119/119 and unknown abstention 31/31.
- Raised `MAX_FEATURES` to 512 and `MAX_VOCAB` to 1024 in the C runtime, Rust port and tools (model structure about 514 KiB).
- Added a std-only Rust port of the runtime (`rust/`) with output parity against the C binary, checked in CI. Bounded parsing replaces fixed C buffers; dangling state references are rejected at load time.
- Added `zerolm live` (ZeroLM Live): persistent decaying memory, rolling transcript for follow-ups, `teach` and good/bad feedback as a learned-reflex overlay, a `--jsonl` daemon mode, and `--tag NAME` sessions with life-file locking so sessions cannot overwrite each other. `weights.zlm` is never modified. See `docs/LIVE.md`.

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
