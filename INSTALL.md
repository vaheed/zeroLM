# ZeroLM CI update

This is a drop-in CI update for `vaheed/zeroLM`.

## Files in this package

Copy these paths into the root of your existing repository, preserving the directory structure:

- `.github/workflows/ci.yml` — expanded Ubuntu/macOS workflow, full pipeline, examples, resource guard, job summary, and uploaded reports.
- `scripts/run_examples.py` — executes documented prompts in one-shot and interactive modes and checks the expected intent plus non-empty output.
- `scripts/ci_summary.py` — writes a readable Markdown report to `$GITHUB_STEP_SUMMARY` and `reports/ci-summary.md`.
- `examples/prompts.tsv` — the small set of prompts used by the example smoke test.

## Apply

From the repository root, extract this archive over the checkout, then:

```bash
git add .github/workflows/ci.yml scripts/run_examples.py scripts/ci_summary.py examples/prompts.tsv
git commit -m "ci: add examples, metrics, and GitHub Actions summary"
git push
```

Or copy the four paths manually from this archive.

## What the updated workflow does

1. Runs on Ubuntu and macOS for pushes to `main`, pull requests, and manual dispatch.
2. Checks the model specification, Python syntax, shell syntax, and generated-weight consistency.
3. Runs the repository's existing `make pipeline` (build, C smoke tests, regression evaluation, weight audit, and benchmark).
4. Runs a separate set of documented CLI examples, including interactive stdin mode.
5. Saves machine-readable example, weight-audit, and benchmark results.
6. On Linux, runs the C smoke tests under a 2 GiB virtual-address-space limit.
7. Writes test counts, model structure, benchmark data, runner information, and caveats to the GitHub Actions job Summary.
8. Uploads the `reports/` directory and compiled binary as a 14-day workflow artifact, even when a prior test step fails.

## Important interpretation

- The current evaluator checks expected intent selection and non-empty response output. It does not establish semantic correctness or general language understanding.
- The prompt set is finite and hand-authored. Passing it is a smoke/regression signal, not a general language benchmark.
- The Linux `ulimit` check limits virtual address space. It is not a physical-RAM measurement or a guarantee for every 2 GB computer.
- Benchmark results include process startup and terminal I/O and should be compared on equivalent runners.
- The workflow does not train, fine-tune, distill, or optimize model weights.

## View the evidence

After a workflow run, open the run's **Summary** tab for the Markdown report. Download `zerolm-ci-reports-Ubuntu` or `zerolm-ci-reports-macOS` from the run's **Artifacts** section for JSON reports and logs.
