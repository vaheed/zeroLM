# Testing and repeatable experiments

## Test layers

### C smoke tests

```sh
./zerolm --self-test
```

These run 20 classifier cases (including two unknown prompts) and then one decoder-path check per intent, 48 checks in total for the current 28 intents. They check that the runtime loads and executes the handwritten network. The Rust binary has the same `--self-test` and prints identical output.

### Regression tests

`tests/regression.tsv` contains labeled English prompts. Each row is:

```text
intent<TAB>prompt
```

The test runner launches the real C binary, reads the selected intent from `--verbose` diagnostics, and checks that a response was emitted.

### Unknown-input tests

`tests/unknown.tsv` contains prompts outside the small supported set. The expected result is `unknown`. These cases measure whether the model abstains instead of guessing.

### Weight graph audit

```sh
python3 scripts/weight_audit.py
```

The audit checks that all states exist, intent starts are valid, transitions point to valid states, decoder paths do not contain cycles, and the file follows the expected format.

### Runtime benchmark

```sh
python3 scripts/benchmark.py --iterations 500
```

The benchmark sends repeated prompts through one interactive process. It includes process startup and terminal I/O, so use it to compare runs on the same computer, compiler, and operating system. It is not a universal hardware guarantee.

### Rust tests

```sh
cd rust
cargo test
cargo clippy --all-targets -- -D warnings
```

- **Unit tests (42)** in the library: parsing and rejection of bad model files, phrase matching,
  decoder equivalence between the pointer walk and the softmax path, memory (overwrite, recall,
  eviction, fading, persistence), the transcript budget, commands, follow-ups, tags, JSON helpers,
  the file lock, the strict JSON parser, and the HTTP request parser.
- **End-to-end tests (16)** in `rust/tests/live_cli.rs` run the real binary against the real
  `weights.zlm` as subprocesses:
  - facts survive a restart; tags are isolated;
  - a second live process on a busy tag is refused, a different tag is not, and the tag is free
    again after the first exits or is `SIGKILL`ed;
  - the JSON-lines contract (plain and JSON input, escapes, unicode, error line, blank lines);
  - teaching and `good`/`bad` feedback persist, and `weights.zlm` stays byte-identical;
  - capacity eviction, memory fading (and recall keeping a fact alive), follow-ups with the real
    model, and that gibberish does not borrow the previous answer;
  - hostile input (a 120 KB line, NUL bytes, emoji, empty commands) does not crash;
  - a corrupt life file and invalid arguments exit with code 2 and leave the file untouched;
  - one-shot mode stays stateless and honours `ZEROLM_MODEL`.

- **API end-to-end tests (19)** in `rust/tests/api.rs` start `zerolm serve` on a random port and speak
  hand-written HTTP over TCP (the 19th checks that an oversized request still gets its 413 while the client keeps streaming):
  - the spec and the route table list exactly the same operations, and the served spec is the file;
  - sessions survive a server restart; tags are isolated; deleting one removes only its file;
  - bearer auth (wrong, short and long tokens, case-insensitive scheme, open health, no route probing);
  - a public bind without a strong token refuses to start;
  - malformed and hostile requests: bad JSON, wrong types, oversized body and headers, chunked bodies,
    repeated `Content-Length`, path-traversal tags, wrong methods (with `Allow`);
  - slow clients are cut off by a total deadline (including a one-byte-at-a-time trickle) without
    blocking others; connection and session caps return 503 and recover;
  - the CLI and the API cannot share a session in either direction, and errors leak no server paths;
  - 8 concurrent clients lose no turns;
  - OpenAI compatibility: model list, completion shape and usage, messages as context, `zerolm-live`
    per `user`, SSE streaming, `max_tokens`, and 23 invalid-request cases that must use OpenAI's error
    envelope.

### API compatibility with real clients

```sh
pip install openai openapi-spec-validator
openapi-spec-validator rust/openapi.yaml
python3 scripts/check_openai_sdk.py HOST:PORT [KEY]   # against a running `zerolm serve`
```

The validator checks `rust/openapi.yaml` against the OpenAPI 3.0 schema (the Rust contract test only
compares operations, so it cannot catch invalid YAML; the validator found one such mistake). The SDK
script runs 15 checks with the official OpenAI Python SDK: models, completions, context, streaming,
`max_tokens`, live sessions, and typed errors.

### C / Rust parity

CI builds both binaries and diffs their output for `--self-test`, `--info`, and `--verbose` on every
prompt in `tests/regression.tsv`, `tests/unknown.tsv` and `examples/prompts.tsv`, against both the
default and `--faithful` Rust decoders. Run it locally the same way: build both, then compare
`./zerolm --verbose "$prompt"` with `rust/target/release/zerolm --verbose "$prompt"`.

### Docker smoke test

```sh
scripts/docker_smoke.sh
```

Builds the image and checks a one-shot answer, memory across containers, tag isolation, JSON-lines
mode, refusal of a second container on a busy tag, and reuse of the tag after the holder is
killed. It also starts the `zerolm-api` service: it must refuse to start without a token, become
healthy, reject requests without a key, answer chat completions through the published port, keep a
session across a restart, and hold that session against a CLI container. It uses its own compose project and removes it afterwards.

## Full pipeline

```sh
make pipeline
```

The pipeline rebuilds the weight file from the specification, compiles C, runs smoke tests, evaluates supported and unknown prompts, audits the graph, and benchmarks inference. Evaluation outputs are written to `reports/latest.json` and `reports/latest.md`.

## Benchmarks

```sh
python3 scripts/benchmark.py --iterations 500   # C runtime, one interactive process
python3 scripts/benchmark_all.py                # C vs Rust vs ZeroLM Live; writes reports/benchmarks.{json,md}
```

`benchmark_all.py` needs both binaries built. It measures:

1. one-shot process time (start-up + answer + exit) and model-load-only time (`--info`);
2. steady-state throughput in one process: C, Rust `--faithful` (same softmax math), and the
   default Rust decoder;
3. peak resident memory of a running process, read from its own `/proc` entry (`wait4`'s
   `ru_maxrss` is not usable: it inherits the parent's size across `fork`);
4. ZeroLM Live lock-step round-trip latency (p50/p95/p99) with the life file on disk and on tmpfs, an
   empty memory and 64 stored facts.

The 64-fact runs keep every fact alive by asking about each in rotation, because unused facts fade
after about 100 turns; the script aborts if fewer than 64 remain at the end. The latest recorded
run is in [reports/benchmarks.md](../reports/benchmarks.md). Its main findings, on one 32-thread Xeon
E5-2697 v4 host:

- C and Rust `--faithful` do the same work at the same speed (about 930 turns/s); the default Rust
  decoder is about 22x faster because it skips the softmax.
- A one-shot call costs 7-13 ms, almost all process start-up and model loading.
- A Live turn costs about 3.4 ms with the life file on disk and about 0.3 ms on tmpfs, so the
  per-turn fsync of the life file dominates; memory size makes no measurable difference.

These are single-run wall-clock numbers on one machine. Compare rows within a report, not across
machines, and re-run on your own hardware.

## How to run a useful improvement cycle

1. Save the current report: `cp reports/latest.json reports/baseline.json`.
2. Add the failing prompt to `tests/regression.tsv` or `tests/unknown.tsv` before editing the model.
3. Inspect the relevant input features and response path with `python3 scripts/inspect_weights.py`.
4. Change one small set of features, weights, or one response in `model/spec.json`, or apply a structured patch.
5. Run `make pipeline` again.
6. Compare results: `python3 scripts/compare_reports.py reports/baseline.json reports/latest.json`.
7. Keep the edit only if the target case is fixed and no measured metric regresses.

Do not use only the current test prompts to claim language quality. Add new prompts from people who did not write the model, including paraphrases and confusing cases. Keep a private holdout set if you want a more meaningful estimate.

## Current baseline

The v0.2.0 test set started at 79 regression prompts and 18 unknown prompts (18 intents). The
current build has 28 intents, 119 regression prompts (including 40 for the ten newer intents) and
31 unknown prompts (including 13 near-misses such as "how do I install Photoshop"), and passes all
150. The Rust port has 27 unit tests and 16 end-to-end tests. These are checks of the included
hand-authored cases only; they are not a general language benchmark.

Known gap: "how much memory does a human brain have" is answered by the `memory` intent, because
the older `memory` features match the word on its own. Phrase matching is exact and context-blind.

### Online benchmark (CLINC150)

```sh
python3 scripts/online_benchmark.py --limit 400
```

Downloads the public CLINC150 intent dataset and reports in-scope accuracy (CLINC intents with a ZeroLM equivalent) and the abstention rate on out-of-scope and unsupported queries. CI fails if abstention drops below 85% or in-scope accuracy below 15%. If the download fails the script skips (use `--require-network` to make that an error).
