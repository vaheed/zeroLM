# Contributing to ZeroLM

Thanks for helping with the experiment. Keep changes small, measurable, and honest.

## Rules

- Keep the shipped model English-only.
- Do not add pretrained weights, training code, fine-tuning, distillation, or gradient descent.
- Do not label test execution as training or learning.
- Do not claim general language understanding from phrase coverage.
- Keep the C runtime dependency-free apart from the standard and math libraries.
- The Rust port (`rust/`) must stay std-only and byte-identical to the C runtime for one-shot prompts (`cargo test` plus the CI parity job).
- Live mode may keep runtime state (facts, learned phrase weights) only in a separate life file. It must never modify `weights.zlm`, and updates must be explicit user input or fixed rules, never gradient or loss based. Call it adaptation or state, not training.
- Every behavior change must include regression or unknown-input test cases.
- New phrase features must be tied to ZeroLM or to "it/this/you" (`install zerolm`, not `install`). Add near-miss prompts to `tests/unknown.tsv` for any broad phrase; bare words cause false positives such as "how do I install Photoshop".

## Development loop

```sh
make pipeline
cp reports/latest.json reports/baseline.json
# Edit model/spec.json or code, and update tests.
make pipeline
python3 scripts/compare_reports.py reports/baseline.json reports/latest.json
```

### Rust and Docker changes

```sh
cd rust
cargo test                                   # unit tests + end-to-end tests of zerolm live
cargo clippy --all-targets -- -D warnings
cd .. && make && (cd rust && cargo build --release)
# the Rust output must match the C runtime on every test prompt; CI diffs them (see docs/TESTING.md)
scripts/docker_smoke.sh                      # if you touched the Dockerfile, compose file or locking
python3 scripts/benchmark_all.py --no-write  # if you touched anything performance-relevant
```

Changes to the HTTP API must update `rust/openapi.yaml` (a test fails if routes and spec differ; validate it with
`openapi-spec-validator`), add a case to `rust/tests/api.rs`, and keep `scripts/check_openai_sdk.py` passing.

Changes to live mode need an end-to-end test in `rust/tests/live_cli.rs`, and the docs in
`docs/LIVE.md` must stay accurate (several bugs so far were found by those tests).

## Editing the model

`model/spec.json` is the source of truth. `build_model.py` serializes it to `weights.zlm`; it never fits or optimizes weights. Prefer one small feature or response change at a time. Keep a backup before experiments.

## Pull requests

A pull request should explain:

1. The exact prompt or failure it addresses.
2. The feature/weight/response change made.
3. The baseline and candidate reports.
4. Any new limitations or possible false matches.

All CI jobs must pass before merge.
