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

## Development loop

```sh
make pipeline
cp reports/latest.json reports/baseline.json
# Edit model/spec.json or code, and update tests.
make pipeline
python3 scripts/compare_reports.py reports/baseline.json reports/latest.json
```

## Editing the model

`model/spec.json` is the source of truth. `build_model.py` serializes it to `weights.zlm`; it never fits or optimizes weights. Prefer one small feature or response change at a time. Keep a backup before experiments.

## Pull requests

A pull request should explain:

1. The exact prompt or failure it addresses.
2. The feature/weight/response change made.
3. The baseline and candidate reports.
4. Any new limitations or possible false matches.

All CI jobs must pass before merge.
