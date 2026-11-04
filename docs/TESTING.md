# Testing and repeatable experiments

## Test layers

### C smoke tests

```sh
./zerolm --self-test
```

These tests cover one representative request for each intent, unknown prompts, and successful traversal of every decoder path. They check that the runtime loads and executes the handwritten network.

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

## Full pipeline

```sh
make pipeline
```

The pipeline rebuilds the weight file from the specification, compiles C, runs smoke tests, evaluates supported and unknown prompts, audits the graph, and benchmarks inference. Evaluation outputs are written to `reports/latest.json` and `reports/latest.md`.

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

The repository's initial v0.2.0 test set contains 79 regression prompts and 18 unknown prompts. The current build passes all 97. This is a check of the included hand-authored cases only; it is not a general language benchmark.
