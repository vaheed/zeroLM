---
name: zerolm-roadmap
description: Use when implementing, reviewing, or planning any ZeroLM phase (0-11): gives each phase's goal, deliverables, CI tests and exit gate from docs/ROADMAP.md.
---

# ZeroLM roadmap skill

Generated from `docs/ROADMAP.md` (source of truth; re-read it for detail). Work phases in order. Before starting a phase: read its section, check `research/phases.json`, run the Phase 0 audit. Before marking it complete: executable command, report schema, CI job with `if: always()` summary, and updated `docs/CLAIMS_AND_LIMITATIONS.md`.

Invariants: see `AGENTS.md` and `docs/RESEARCH_CONSTITUTION.md`.

## Phase 0 — Research constitution and scope freeze

**Goal:** Turn the updated objective into an unambiguous scientific contract.

**Deliverables**
- `docs/RESEARCH_CONSTITUTION.md`
- `docs/CLAIMS_AND_LIMITATIONS.md`
- `research/manifest/seed-knowledge.json`
- `research/manifest/update-policy.json`

**CI tests**
- Static scan for forbidden pretrained artifacts and model downloads.
- Validate seed-knowledge manifest and update-policy schema.
- Ensure every weight group declares its initialization source and permitted update path.
- Fail if a research phase is marked complete without an executable test command and report schema.

**Exit gate**
- A reviewer can determine exactly what is manually programmed, what is learned online, and what is prohibited.
- No claims of “no learning” remain for the online-learning branch.

## Phase 1 — Freeze and audit the existing baseline

> **Implemented differently:** the legacy code was deleted, so Phase 1 froze the new baseline (`zerolm-baseline`, `docs/BASELINE.md`, tag `legacy-final`). Keep `zerolm-baseline check` green.

**Goal:** Keep the current work reproducible and prevent the new engine from hiding regressions.

**Deliverables**
- `reports/baseline/legacy.json`
- `reports/baseline/next.json`
- `reports/baseline/environment.json`
- `docs/BASELINE.md`

**CI tests**
- `make pipeline`
- `python3 research/build_next.py --check`
- `cargo test --manifest-path rust/Cargo.toml`
- `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings`
- Existing v0.2 C/Rust parity, Next benchmark, ablation, and patch-validation suites.

**Exit gate**
- Clean build and full baseline CI pass.
- Commit SHA, weight SHA-256, benchmark seed, environment, and dirty-tree state are recorded.

## Phase 2 — Neural compute engine and architecture bake-off

> **Implemented:** crates tensor/nn/tokenizer/engine/eval; GRU selected (`model/specs/architecture.json`). Keep `zerolm-bakeoff --check` green; the f64 reference lives in Rust tests (no Python).

**Goal:** Create a reusable neural runtime rather than another specialized parser.

**Deliverables**


**CI tests**
- Compare Rust kernels against a small independent Python reference on seeded inputs.
- Numerical parity: `max_abs_error <= 1e-6` for FP32 reference tests, with explicit tolerances for any lower-precision kernel.
- Determinism test: same seed and spec produce byte-identical initialized weights.
- Recurrent equivalence: token-by-token and chunked recurrence produce matching final states where mathematically expected.
- Ablations: zeroing/shuffling key recurrent paths must damage the tasks that depend on them.
- CPU benchmark: report p50/p95 token latency, tokens/sec, peak RSS, parameter count, and checkpoint size.

**Exit gate**
- One recurrent candidate is selected by measured trade-offs, not preference.
- The architecture can process long streams with fixed-size session state and no growing KV cache.
- A clean CPU-only build succeeds on Linux and macOS.

## Phase 3 — Bootstrap language interface and deterministic initialization

**Goal:** Make the system able to receive arbitrary text before it has learned a vocabulary.

**Deliverables**
- `crates/zerolm-tokenizer/`
- `crates/zerolm-engine/src/bootstrap.rs`
- `model/bootstrap/seed.json`
- `docs/BOOTSTRAP.md`

**CI tests**
- Round-trip UTF-8 tests for ASCII, Unicode, punctuation, emoji, and malformed input handling.
- No-OOV test: every valid UTF-8 string is representable.
- Deterministic initialization hash test.
- Static artifact scan proving no pretrained tokenizer or embedding asset is loaded.
- CLI test: a clean model starts, accepts the teaching protocol, emits an inspectable trace, and exits without corrupting weights.

**Exit gate**
- The model can accept arbitrary byte sequences from the first run.
- It is explicitly not claimed to understand arbitrary English at bootstrap.

## Phase 4 — First true online neural language model

**Goal:** Demonstrate online language acquisition rather than phrase lookup or fixed answers.

**Deliverables**
- `crates/zerolm-learning/src/online_predictor.rs`
- `research/streams/`
- `research/benchmark/language_modeling.py`
- `reports/online-learning/`

**CI tests**
- Start from deterministic initial weights and verify initial and final hashes differ after a real update.
- Verify update event log corresponds exactly to observed tokens and prediction targets.
- Verify no future token leaks into the prediction state.
- Verify no update occurs when learning mode is disabled.
- Evaluate held-out continuation and novel combinations never included in the exposure stream.
- Run at least three seeds and report mean, standard deviation, and per-seed curves.
- Require statistically/reproducibly better held-out BPB than the frozen initialized model and the byte-frequency baseline before claiming acquisition. Set the numeric improvement floor after Phase 2 pilot baselines and freeze it before the main experiment.
- Report the online-BPTT control separately; it must never be presented as the strict local-plasticity result.

**Exit gate**
- The same initially untrained architecture improves on unseen continuations after chronological exposure.
- Improvement is not explained only by memorizing seen strings.
- The update rule and its limitations are explicit. If non-gradient plasticity cannot pass, do not hide the failure: either revise the rule or explicitly decide whether online gradient updates are allowed.

## Phase 5 — Weight-resident associative memory

**Goal:** Give the neural model persistent memory without transcript replay or an external semantic database.

**Deliverables**
- `crates/zerolm-memory/src/fast_weights.rs`
- `crates/zerolm-memory/src/persistent_memory.rs`
- `crates/zerolm-memory/src/retrieval.rs`
- `docs/MEMORY_FORMAT.md`

**CI tests**
- Add at least 1,000 synthetic facts through the live update API, then restart from the checkpoint and retrieve them.
- Exact key recall and paraphrase retrieval are scored separately.
- Test collisions, capacity, interference, contradictory facts, supersession, deletion, and memory compaction.
- Prove that semantic facts are present in checkpoint tensor groups and not in `life.json`, transcript files, SQLite, or a vector database.
- Verify that unrelated `W_CORE` tensors remain byte-identical during a memory-only write.
- Corrupt-checkpoint and interrupted-write tests must fail safely or recover the last valid checkpoint.

**Exit gate**
- Persistent recall works across process restarts using only model checkpoint tensors and the model's retrieval computation.
- Memory capacity, retrieval precision/recall, latency, and interference curves are published.

## Phase 6 — Concepts, relations, and compositional reasoning

**Goal:** Move beyond token prediction and exact fact recall to reusable internal concepts and inference.

**Deliverables**
- `crates/zerolm-engine/src/semantic_state.rs`
- `crates/zerolm-engine/src/binding.rs`
- `crates/zerolm-reasoner/`
- `research/benchmark/compositional_reasoning.py`

**CI tests**
- Teach a set of primitive relations, then test unseen combinations and paraphrases.
- Test counterfactual pairs: changing one premise must change the inference when logically required.
- Test contradiction detection and abstention when premises are insufficient.
- Compare against a symbolic reference parser/reasoner and a lookup baseline.
- Run causal ablations of binding, recurrent state, retrieval, and each reasoning operator.
- Use hidden test generation with a seed and publish the seed/hash, not the private test labels.

**Exit gate**
- The model answers new compositional questions from stored primitives and relations, not from a full-sentence lookup.
- Causal ablations demonstrate that the relevant neural pathways are actually used.

## Phase 7 — Grounded learning and interaction

**Goal:** Give concepts meaning through observed consequences, not language strings alone.

**Deliverables**
- `crates/zerolm-world/`
- `research/environments/`
- `research/benchmark/grounding.py`
- `docs/GROUNDED_LEARNING.md`

**CI tests**
- Prediction-before-observation tests: the model must commit to an outcome before seeing the result.
- Intervention tests: change one variable and verify counterfactual predictions change appropriately.
- Action-result learning: performance must improve over a frozen model on unseen combinations.
- Test transfer to new object names, properties, and action sequences.
- Verify the environment result, not the model's own text, is the learning target.

**Exit gate**
- Demonstrable forward transfer from learned experiences to new tasks in a controlled world.
- No claim of physical-world understanding until multimodal grounding has been independently tested.

## Phase 8 — Natural-language generation and multi-turn dialogue

**Goal:** Produce new language from the model's learned internal state.

**Deliverables**
- `crates/zerolm-engine/src/decoder.rs`
- `crates/zerolm-runtime/src/dialogue.rs`
- `research/benchmark/dialogue.py`

**CI tests**
- Generate continuations not present in the exposure stream and not encoded as templates.
- Evaluate response consistency with retrieved facts and reasoning traces.
- Multi-turn tests must survive process restart when the relevant information has been stored in `W_MEMORY`.
- Test hallucination, unsupported questions, calibration, repetition loops, and prompt paraphrases.
- Measure free-generation diversity and degeneration; do not score only exact strings.

**Exit gate**
- The system is a genuine autoregressive neural language model with learned predictive behavior and non-template generation.
- It remains honest about uncertainty and does not present unverified self-generated text as fact.

## Phase 9 — Controlled self-modification and internal WSP

**Goal:** Enable the model to improve the structure of its own neural engine safely.

**Deliverables**
- `crates/zerolm-wsp/src/patch.rs`
- `crates/zerolm-wsp/src/validator.rs`
- `crates/zerolm-wsp/src/sandbox.rs`
- `crates/zerolm-wsp/src/rollback.rs`
- `docs/WSP_PROTOCOL.md`

**CI tests**
- Schema, tensor shape, value range, and protected-region validation.
- Candidate patch is applied to a copy; active checkpoint hash remains unchanged until acceptance.
- Target tests, all frozen regression suites, adversarial tests, and CPU/RAM budgets run before acceptance.
- Fault injection: malformed patch, NaN, out-of-range edit, failed test, interrupted checkpoint, and rollback.
- Compare the patch against a no-op patch, random patch, and manually authored reference patch.
- Report target gain, regression delta, changed-parameter count, tensor names, and full before/after hashes.
- No patch may edit the tests, evaluator, workflow, or protected manifest.

**Exit gate**
- The engine can propose at least one useful patch from a trace and evidence; the patch passes a hidden evaluation and does not regress protected tasks.
- If it can only apply prewritten patch templates, describe it honestly as rule-based self-maintenance, not autonomous WSP.

## Phase 10 — Developmental capability evaluation

**Goal:** Measure whether the system is becoming more capable over a sequence of experiences.

**Deliverables**
- `research/benchmark/developmental_suite/`
- `reports/developmental/learning_curves.json`
- `reports/developmental/forgetting.json`
- `docs/DEVELOPMENTAL_EVALUATION.md`

**CI tests**
- Report per-stage performance and learning curves.
- Report average accuracy, forgetting, backward transfer, forward transfer, calibration, memory capacity, and compute cost.
- Require performance improvement on new tasks while bounding regressions on protected old tasks.
- Run multiple seeds and distinguish independent test data from exposure streams.
- Detect benchmark contamination by content hashes and source manifests.

**Exit gate**
- Publish a capability profile with both successes and failures. “Equivalent to a seven-year-old” is not an allowed release claim unless a suitable independent, normed evaluation actually supports that exact statement.

## Phase 11 — CPU deployment, checkpoint durability, and 1.0 release

**Goal:** Turn the validated research engine into a usable local model.

**Deliverables**


**CI tests**
- Linux and macOS CPU builds; x86_64 first, ARM64 when runner availability permits.
- Offline execution test: no network access during inference or learning.
- Restart and checkpoint migration tests.
- Resource benchmark on a named, documented runner profile; report p50/p95 latency, tokens/sec, peak RSS, checkpoint size, and update cost.
- CLI/API end-to-end tests and Docker smoke test.
- Full regression and developmental suite.

**Exit gate**
- Reproducible CPU-only release, persistent weight-resident memory, online learning, safe WSP updates, and documented capability boundaries.
