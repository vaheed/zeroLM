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

## ZeroLM Next job (`next`)

A separate job in `.github/workflows/ci.yml` runs the research track on ubuntu-latest. It records the commit, toolchain versions and the benchmark seed, then: validates `model/next/spec.json` and every patch; **reproduces** `model/next/weights.zln` from the specification and patches (`research/build_next.py --check`); builds the C reference and the Rust runtime; runs `cargo test` (including the shortcut-detection and causal-dependence tests in `rust/tests/next.rs`) and clippy; checks that the legacy v0.2 regression suite is still 100 % and that C and Rust agree; checks that every weight patch validates and has recorded results; runs the development, adversarial, primitive and per-defect suites (each must pass in full); regenerates the compositional holdout from its seed and verifies its sha256 against `tests/next/heldout_manifest.json`; then runs `research/benchmark/run_suite.py`, which evaluates ZeroLM Next, the symbolic, keyword, phrase-lookup and legacy baselines, random and shuffled weights, 28 ablations and a length-scaling sweep. The job fails if an acceptance floor is missed or if an ablation of a designed mechanism has no measurable effect (a sign the network is being bypassed). Machine-readable reports and execution traces are uploaded as the `zerolm-next-reports` artifact and a readable summary is written to the GitHub Actions job summary.

Nothing is marked as passed when skipped: every step either runs to completion or the job fails.
