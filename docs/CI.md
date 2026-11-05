# Continuous integration

Workflow: `.github/workflows/ci.yml`. It runs on pushes to `main`, pull requests, and manual
dispatch, with read-only repository permissions. It never trains, fine-tunes, distills or
optimizes weights.

## Jobs

### `test` — build, test and examples (Ubuntu and macOS)

1. Validates `model/spec.json`, Python syntax, shell syntax, and that `weights.zlm` matches
   what `build_model.py` generates from the spec.
2. Runs `make pipeline`: build, C smoke tests, regression and unknown-input evaluation,
   weight audit, benchmark.
3. Runs the documented CLI examples, including interactive stdin mode.
4. Saves machine-readable weight-audit and benchmark results.
5. Runs the online CLINC150 benchmark (needs network; skipped if the download fails).
6. On Linux, runs the C smoke tests under a 2 GiB virtual-address-space limit.
7. Writes a Markdown summary to the job's Summary tab and uploads `reports/` and the binary as
   an artifact (14 days), even if an earlier step failed.

### `rust` — Rust port (Ubuntu)

1. `cargo test`: unit tests plus the end-to-end tests of `zerolm live` in `rust/tests/`.
2. `cargo clippy --all-targets -- -D warnings`.
3. Builds the C reference and the Rust release binary and **diffs their output** (`--self-test`,
   `--info`, and `--verbose` for every prompt in `tests/` and `examples/`), in both the default
   and `--faithful` decoder modes.
4. Smoke-tests live mode: a fact and a taught phrase survive a restart, and `weights.zlm` has the
   same checksum afterwards.
5. Starts `zerolm serve` and runs `scripts/check_openai_sdk.py` (official OpenAI Python SDK, installed with
   pip), after validating `rust/openapi.yaml` with `openapi-spec-validator`.

### `docker` — container image (Ubuntu)

Runs `scripts/docker_smoke.sh`: builds the image, then checks a one-shot answer, memory across
containers, tag isolation, JSON-lines mode, refusal of a second container on a busy tag, and
reuse of the tag after the holder is killed. It also checks the `zerolm-api` service: it refuses to start
without a token, passes its health check, rejects unauthenticated requests, serves chat completions
on the published port, persists a session across a restart, and holds that session against a CLI container.

## Reading the results

- The evaluator checks expected-intent selection and non-empty output. It does not establish
  semantic correctness or general language understanding.
- The prompt set is finite and hand-authored: passing it is a regression signal, not a language
  benchmark.
- The `ulimit` check limits virtual address space. It is not a physical-RAM measurement.
- Benchmark results include process start-up and should only be compared on equivalent runners.
  For a fuller, reproducible benchmark run `python3 scripts/benchmark_all.py` (see
  [TESTING.md](TESTING.md)).

## Evidence

Open a run's **Summary** tab for the Markdown report. Download `zerolm-ci-reports-Linux` or
`zerolm-ci-reports-macOS` from **Artifacts** for the JSON reports and logs.
