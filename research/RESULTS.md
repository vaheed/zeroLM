# Results

All numbers below are copied from `reports/next/*.json` (regenerate with `python3 research/benchmark/run_suite.py`). Environment: Linux x86-64, Rust 1.75, GCC 13.3, Python 3.12, one CPU thread. The reports record the git commit, `git_dirty`, seeds and sha256 of weights, cases and generator. **Provenance caveat:** the work is not yet committed, so the recorded commit is the pre-change HEAD and `git_dirty` is `true`.

## 1. Protocol

| set | seed | cases | role |
|---|---:|---:|---|
| `dev.jsonl` | 1001 | 527 | development; restricted lexical subset, ≤ 3 clauses. The only data inspected while the base construction was designed |
| holdout "discovery" | 20260601 | 21,585 | evaluated once on the base construction; exposed defects 001/002; development data afterwards |
| holdout "gate" | 20260615 | 21,585 | regression gate for patch 004 and an intermediate check; development data afterwards |
| **holdout "reported"** | **20260701** | **21,585** | never inspected before the patch set was frozen. All headline numbers |
| `adversarial.jsonl` | hand-written | 72 | boundary-of-grammar probes (15 accept, 57 reject) |

The reported holdout has 19,785 supported and 1,800 unsupported/ambiguous cases, including 13,500 compositional cases (single 2,000; two clauses 3,000; three 3,000; 4-5 clauses 2,000; 6-12 clauses 1,500; binding 1,000; state tracking 1,000), 3,000 order cases (600 groups × 5 connector forms), 3,200 counterfactual cases (1,600 pairs), 85 primitive cases, 1,500 unsupported and 300 ambiguous cases. "Seen" means the exact operation sequence occurs in `dev.jsonl`.

## 2. Headline: ZeroLM Next (final weights, patches 001-004) on the reported holdout

| metric | target | result | cases |
|---|---:|---:|---:|
| supported primitive interpretation | ≥ 99 % | **100 %** | 85 |
| held-out compositional exact match (unseen sequences) | ≥ 95 % | **100 %** | 13,294 |
| ... unseen primitive combinations (at least one operation never in dev) | | 100 % | 13,021 |
| ... seen sequences | | 100 % | 206 |
| ... longer than dev (> 3 operations) | | 100 % | 4,136 |
| multi-operation execution (final world state) | ≥ 90 % | **100 %** | 14,512 |
| counterfactual pair consistency | ≥ 95 % | **100 %** | 1,600 pairs |
| order groups with all 5 connector forms right | | 100 % | 600 |
| unsupported-input rejection | ≥ 99 % | **100 %** (false acceptance 0) | 1,800 |
| false rejection of supported input | | 0 % | 19,785 |
| per-operation accuracy / semantic accuracy | | 100 % / 100 % | |
| Rust executor vs independent Python executor | | agree on 19,785/19,785 | |
| hand-written adversarial suite | | 72/72 | 72 |
| legacy v0.2 regression suite | 100 % | **150/150** (and 47 + 20 + 16 existing Rust tests pass; C and Rust output identical) | |
| predefined full-sentence lookup / hidden training / pretrained weights | 0 | 0 / 0 / 0 (tests below) | |

**Read this table with the caveat in `HYPOTHESES.md`:** the cases are composed from the grammar the network was built for. 100 % shows the construction implements that grammar, including on combinations absent from dev. It is not evidence about language outside the grammar. The same file is parsed identically (100 %) by an independent symbolic parser (section 4).

## 3. The failure that came first (negative result, kept)

On the discovery holdout the base construction scored:

| metric | base construction | target |
|---|---:|---:|
| compositional exact match | **94.4 %** | ≥ 95 % (**missed**) |
| exact match, all supported | 96.2 % | |
| false acceptance (1,800 unsupported/ambiguous) | 16.7 % (all 300 ambiguous accepted) | ≤ 1 % (**missed**) |
| longer than dev (> 3 operations) | 86.2 % | |
| seen sequences | 100 % | |
| primitives, counterfactual pairs, order groups | 100 % | |

Failures were not spread out. All 759 supported-case failures had one cause: `.` and `;` did not end a row, so a following `after`, `before`, `if` or `unless` was attached to the previous clause (`diagnose` shows `bnd_AFTER` firing at the token after `;`). The error grows with length because long sentences contain more full stops. The same stage evaluated on the reported holdout (not used for any decision) gives 96.3 % exact, 94.6 % compositional, 16.7 % false acceptance: the discovery numbers were not a fluke of one seed (`reports/next/patch_stages.json`).

## 4. Manual weight patches (no optimizer)

`wsp.py diagnose` located each defect in the trace; the fix was reasoned, written as a patch, validated, applied with a backup, and compared to the unchanged baseline on target and regression suites. Full records, including hypothesis and mathematical effect, are in `experiments/next-patches/`.

| patch | defect found by | target suite, before → after | regression suites |
|---|---|---|---|
| 001 full stop / `;` closes the row (new `STOP` token, `OPEN` field, 2 boundary units, 2 write units) | discovery holdout | 759 cases: 0 % → **100 %** exact | dev 100 % → 100 % |
| 002 reject chains `A after B before C` (new `GRP` field, 4 units) | discovery holdout | 300 cases: false acceptance 100 % → **0 %** | dev, 001 targets unchanged |
| 003 remove `that` from the neutral words (`A, after that B` was silently **reversed**) | hand-written adversarial suite | 150 cases: false acceptance 100 % → **0 %** | dev, 001, 002 unchanged |
| 004 a connector must be followed by a clause (new `PEND` field, 4 units) | adversarial suite | 150 cases: false acceptance 85.3 % → **0 %** | dev, 001-003 targets and the 21,585-case gate holdout unchanged |
| (runtime) tokenizer: unknown symbols such as emoji become unknown tokens instead of being dropped | adversarial suite | `move the red block left 💥` accepted → rejected | `cargo test` |

Per-stage results on the reported holdout (`reports/next/patch_stages.json`; the runtime tokenizer fix applies to all stages):

| stage | exact match | compositional | false acceptance (holdout) | adversarial false acceptance |
|---|---:|---:|---:|---:|
| base | 96.3 % | 94.6 % | 16.7 % | 3.5 % |
| + 001 | 100 % | 100 % | 16.7 % | **7.0 %** (new: `A. then` accepted) |
| + 002 | 100 % | 100 % | 0 % | 7.0 % |
| + 003 | 100 % | 100 % | 0 % | 3.5 % |
| + 004 | 100 % | 100 % | 0 % | 0 % |

Patch 001 *introduced* a new false acceptance (a connector after a full stop was no longer counted as dangling) that patch 004 repaired. The patch workflow also auto-reverted a patch once during development, because of a bug in the workflow's own improvement check (a metric of 0.0 was treated as missing); the bug was fixed and the patch re-applied.

## 5. Baselines (reported holdout, 21,585 cases)

| system | exact match | per-op | semantic | false acceptance | false rejection |
|---|---:|---:|---:|---:|---:|
| **ZeroLM Next** | **1.0000** | 1.0000 | 1.0000 | 0 | 0 |
| symbolic recursive-descent parser (`research/baselines/symbolic.py`) | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| keyword intent classifier (one operation from the first keywords) | 0.2307 | 0.1971 | 0.2629 | 0.5644 | 0.0050 |
| phrase lookup of the dev sentences | 0.0012 | 0.0004 | 0.0012 | 0 | 0.9988 |
| legacy v0.2 (C runtime) | 0 | 0 | 0 | 0.0067 | 1.0 |
| random weights, same architecture (5 seeds) | 0 (max) | | ≤ 0.074 | 0.0 - 0.99 | |
| same weight values, shuffled (5 seeds) | 0 (max) | | 0 | 0 | |

* The symbolic parser ties with the network. **H6 (network beats symbolic parser) is a negative result**; the claim is parity with full inspectability and no training, not superiority. The first version of the baseline scored 99.0 % because it shared the network's `. if` ambiguity; it was a bug in the baseline and was fixed before this comparison, so that the network is not flattered.
* The legacy v0.2 model is an intent classifier that cannot output operations. It rejects 99.9 % of the inputs and produces no valid output: it is reported for completeness, not as a skill comparison.
* The phrase-lookup baseline shows what a table of seen sentences achieves on unseen combinations: 0.1 %.
* Random-weight semantic accuracy up to 7.4 % comes from networks that accept everything and emit no operation: 7.4 % of cases have gold operations that change nothing (negated or condition-skipped). A do-nothing system gets that for free, so semantic accuracy should be read against it.

## 6. Ablations (reported holdout; reference 100 % exact, 0 % false acceptance)

| ablation | exact | false acceptance | | ablation | exact | false acceptance |
|---|---:|---:|---|---|---:|---:|
| reset state every token (`recurrence`) | 0 | 0 | | zero `B_w` (write layer) | 0 | 0 |
| no reading of the state (`state_read`) | 0.038 | 0 | | zero `B_o` | 0 | 0 |
| uniform pointer (`binding`) | 0 | 0 | | zero `E` (lexicon) | 0 | 0 |
| pointer never advances (`pointer_shift`) | 0.267 | 0.036 | | zero `S` | 0.216 | 0 |
| no time written (`time_order`) | 0.357 (semantic 0.875) | 0 | | zero `A_w` / `A_o` | 0.267 | 0.036 |
| permute rows of `E` | 0 | 0.001 | | zero `V_w` / `V_o` (validity) | **1.0** | **0.319** |
| permute rows of `B_o` | 0 | 0.007 | | zero `B_e` (error units) | **1.0** | **0.348** |
| permute rows of `A_w` | 0.530 | 0 | | zero `Gm_w` (max-time update) | 0.944 | 0 |
| permute rows of `B_w` | 0 | 0 | | permute rows of `S` | 0.941 | 0 |
| permute rows of `V_w` | 0 | 0.082 | | zero `tf_w`, `rank_eps` | 1.0 | 0 |
| permute rows of `dec.1.w` | 0.106 | 0 | | shuffle all values (seed 1) / random Gaussian (seed 1) | 0 / 0 | 0 / 0.988 |

Every designed mechanism the architecture relies on is causal: removing recurrence, binding, pointer advance or time destroys order- and multi-operation behaviour; the lexicon, write and boundary layers are individually necessary; the validity layer and the error units do not affect accuracy on supported input but are what produce rejection (false acceptance rises to 32-35 % without them). **Inert tensors** (zeroing changes nothing observable): `tf_w` (made redundant by patch 004), `rank_eps` (tie-break that valid parses never need), `Gm_b` (an all-zero tensor) and the ten decoder bias vectors (tie-breaks). `cargo test every_weight_tensor_is_causally_used` pins exactly this set, so a new inert tensor, or a tensor that stops mattering, fails the build.

## 7. Latency, memory, scaling (release build, single thread)

| operations in the input | exact match | mean parse µs | p99 µs |
|---:|---:|---:|---:|
| 1 | 100 % | 200 | 364 |
| 4 | 100 % | 758 | 1,306 |
| 8 | 100 % | 1,902 | 3,426 |
| 16 | 100 % | 3,480 | 7,081 |
| 23 | 100 % | 5,931 | 15,272 |
| 24 | 47.7 % | 3,990 | 8,174 |
| 25, 28 | 0 % (rejected) | | |

Timings vary by up to ~2x between runs on this shared machine. Reported holdout, whole batch: mean 403 µs per parse (p50 331 µs, p99 1.54 ms), peak RSS 3.7 MB, weight file 56,444 bytes (24,814 slots, 1,469 non-zero). Cost grows roughly linearly with length (a dense implementation: each token multiplies 11 × 86 and 68 × 140 matrices, each row of the final tape goes through a 26 × 53 matrix, and the 24 × 51 tape is touched). The 24-operation row is a capacity effect, not a mis-parse: a trailing `.`, `;` or connector opens a row, so with 24 rows the maximum is **23 operations when the text ends in punctuation, 24 otherwise**; excess input is rejected by the overflow detector, never mis-parsed. The legacy v0.2 CLI takes 2.5 ms per interaction including process start (`make benchmark`), which is not comparable (it spawns a process per query).

## 8. Is it a neural network, and does it bypass its weights?

* `zerolm next trace` exposes, per token: the embedding, the unbound row, the boundary and write unit activations (named), the write vector, pointer row, running max time and error added; then per row the validity errors, rank and time, and per output slot the readout logits. 8 sample traces are in `reports/next/traces` in CI.
* Tests (`rust/tests/next.rs`, all passing): the 864 sentences `move the {color} {shape} {dir} {n} times` equal the expected operations built only from the loop variables; no multi-word vocabulary entry or response string exists in the weight file; none of the encoder/network/binding/decoder/model/trace source files contains a lexicon word as a string literal; the module has no network or process calls; every tensor except the documented inert set is causally used; ablations and random weights change the probes; the executor API takes `(World, &[Op])` and no text.
* These tests detect the *listed* shortcuts. They cannot prove that no semantics is hidden in the generic code; that rests on reading `rust/src/next/` (≈ 1,200 lines including the CLI and executor), which is short enough to audit.

## 9. Decision on the first hypothesis

**H1 is supported for the bounded grammar, with an important qualification.**

* Supported: with no training, hand-derived weights implemented a compositional parser that scored 100 % on 19,785 supported and 1,800 unsupported held-out cases, including 13,294 never-seen operation sequences, sequences up to 23 operations (dev had ≤ 3), order, negation, conditions, binding and state tracking; ablations and random weights show the behaviour comes from the designed weights; defects were found from traces and repaired by documented manual edits.
* Qualified: the first held-out run **failed** two of the targets (94.4 % compositional, 16.7 % false acceptance), and the generated benchmark could not see three further defects that a 72-case hand-written suite found. The final 100 % is a measurement after four repairs, on a fresh seed; the language it holds for is the one defined in `research/README.md`. An independent symbolic parser does equally well; the network contributes inspectability and the demonstration that this can be done with weights alone, not better coverage.
* Not supported / not tested: anything about open-vocabulary language, learning, or LLM-like generation.

## 10. What was not done

* No remote GitHub Actions run exists; the `next` job was written and each of its steps was executed locally. CI artifact links therefore do not exist yet.
* No HTTP/API entry point for ZeroLM Next (the model contract is not stable). The CLI covers parse, run, trace, batch and interactive use.
* The work is uncommitted in the working tree.
