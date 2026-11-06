# Baseline

Phase 1 freezes what exists so the neural engine cannot hide regressions.

## What the baseline is
- **Current baseline:** the Phase 0 contract (manifests, weight groups, phase gates, forbidden-artifact scan), recorded in `reports/baseline/next.json` as a passing `zerolm-audit` result.
- **Legacy baseline:** the v0.2 C/Rust parser, ZeroLM Next and ZeroLM Live were deliberately deleted (no backward compatibility). They are archived only in git history at tag `legacy-final` (`65ff58bd`), described in `research/baseline/legacy-archive.json`. No numbers from them carry over, so there are no stale caveats to reconcile and no ablations or parser comparisons to keep.

## Reproduce from a clean checkout
```bash
git clone <repo> && cd zerolm
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p zerolm-policy --bin zerolm-baseline -- generate
cargo run -p zerolm-policy --bin zerolm-baseline -- check
```

`generate` writes `reports/baseline/{environment,legacy,next}.json`. `check` fails if any report is missing, unparsable, records `git_dirty=true`, or if the baseline audit did not pass. A missing report is a failure, never a pass. `reports/` is git-ignored; CI uploads it as an artifact.

## Provenance fields
Commit SHA, `git_dirty` (source changes only; generated `reports/` is excluded), OS, CPU architecture and model, core count, rustc and cargo versions.

## Regression rule
Every later phase must keep `cargo test --workspace`, clippy, `zerolm-audit`, and `zerolm-baseline check` green.
