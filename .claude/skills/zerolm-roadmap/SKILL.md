---
name: zerolm-roadmap
description: Use when implementing, reviewing, or planning any ZeroLM phase (0-16): gives each phase's goal, deliverables, CI tests and exit gate from docs/roadmap.md.
---

# ZeroLM roadmap skill

Generated from `docs/roadmap.md` (source of truth; re-read it for detail). Work phases in order. Before starting a phase: read its section, check `research/phases.json`, run the Phase 0 audit. Before marking it complete: executable command, report schema, CI job with `if: always()` summary, and updated `docs/claims-and-limitations.md`.

Invariants: see `AGENTS.md` and `docs/research-constitution.md`.

## Phase 0 — Research constitution and scope freeze

**Goal:** Turn the updated objective into an unambiguous scientific contract.

**Deliverables**
- `docs/research-constitution.md`
- `docs/claims-and-limitations.md`
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

> **Implemented differently:** the legacy code was deleted, so Phase 1 froze the new baseline (`zerolm-baseline`, `docs/baseline.md`, tag `legacy-final`). Keep `zerolm-baseline check` green.

**Goal:** Keep the current work reproducible and prevent the new engine from hiding regressions.

**Deliverables**
- `reports/baseline/legacy.json`
- `reports/baseline/next.json`
- `reports/baseline/environment.json`
- `docs/baseline.md`

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

> **Implemented:** tokenizer hardening, `zerolm-engine/src/bootstrap.rs`, `model/bootstrap/seed.json` (pinned init hash), `zerolm-runtime` CLI, `docs/bootstrap.md`. The ledger is scaffolding, not memory.

**Goal:** Make the system able to receive arbitrary text before it has learned a vocabulary.

**Deliverables**
- `crates/zerolm-tokenizer/`
- `crates/zerolm-engine/src/bootstrap.rs`
- `model/bootstrap/seed.json`
- `docs/bootstrap.md`

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

> **Implemented:** `crates/zerolm-learning` (online predictor; frozen/delta/hebbian rules; streams; n-gram baselines), `research/streams/compositional-v1.json`, frozen protocol `model/specs/online-learning.json`, benchmark `zerolm-online-lm` (Rust, not `language_modeling.py`: the project has no Python), `docs/online-learning.md`. Only the readout learns; BPTT control not run. Tune only on pilot seeds; keep the gate green.

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

> **Implemented:** `crates/zerolm-memory` (`checkpoint.rs` atomic format + recovery, `encoder.rs`, `retrieval.rs`, `fast_weights.rs`, `persistent_memory.rs`), frozen spec `model/specs/memory.json`, benchmark `zerolm-memory` (Rust, no Python), CLI `zerolm memory --dir`, `docs/memory-format.md`. Retrieval is programmed (pooled attention + per-word coverage verification), not learned. Tune only on pilot seeds; keep the gate green.

**Goal:** Give the neural model persistent memory without transcript replay or an external semantic database.

**Deliverables**
- `crates/zerolm-memory/src/fast_weights.rs`
- `crates/zerolm-memory/src/persistent_memory.rs`
- `crates/zerolm-memory/src/retrieval.rs`
- `docs/memory-format.md`

**CI tests**
- Add at least 1,000 synthetic facts through the live update API, then restart from the checkpoint and retrieve them.
- Exact key recall and paraphrase retrieval are scored separately.
- Test collisions, capacity, interference, contradictory facts, supersession, deletion, and memory compaction.
- Prove that semantic facts are present in checkpoint tensor groups and not in `life.json`, transcript files, SQLite, or a vector database.
- Verify that unrelated core-weight tensors remain byte-identical during a memory-only write.
- Corrupt-checkpoint and interrupted-write tests must fail safely or recover the last valid checkpoint.

**Exit gate**
- Persistent recall works across process restarts using only model checkpoint tensors and the model's retrieval computation.
- Memory capacity, retrieval precision/recall, latency, and interference curves are published.

## Phase 6 — Concepts, relations, and compositional reasoning

> **Implemented:** `zerolm-engine/src/binding.rs` (FHRR phasors, fractional-power numbers), `semantic_state.rs` (bounded grammar, pronoun state), `crates/zerolm-reasoner` (relational tensor memory, 8 operators, `oracle.rs`, `lookup.rs`), frozen spec `model/specs/reasoning.json`, benchmark `zerolm-reasoning` (Rust, not `compositional_reasoning.py`), CLI `zerolm reason`, `docs/reasoning.md`. Operators are programmed; the oracle is evaluation-only. Hidden seed from `ZEROLM_HIDDEN_SEED` or the commit. Tune only on pilot seeds.

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

## Revision 2 (phases 7-16) — read docs/roadmap.md sec. 0.3, 3.7, 5, 7.1 first

Thesis: one neural engine (neural ISA); WSP compiles algorithms (never language data) into program weights; local online rules learn from licensed human-written text; 1.0 = Level L3 small LLM. New invariants 10-13 in the constitution: uniform execution (reference-interpreter test), byte-permutation test for circuits, corpus manifest / no LM-generated text, the Gate A decision is the only way to change the no-backprop rule. Revision 3 (roadmap sec. 0.5): Gate A is the early representation-learning decision (Phase 9); the old Phases 9-10 merge into Phase 10 (Gate B).

| Phase | Goal | Exit gate |
|---|---|---|
| 7 Uniform engine + WSP compiler | ISA interpreter, `zerolm-wsp` compiler/patches, recompile Phase 5-6 into weights, delete Rust control flow | Phase 5-6 gates pass through compiled programs; independent interpreter reproduces outputs |
| 8 Real-text streams + baselines | `zerolm-data`, corpus manifest, contamination check, PPM/KN-5, online-backprop CONTROL, BLiMP/LAMBADA/BPB harnesses | Reproducible pipeline; `model/specs/levels.json` frozen |
| 9 Representation feasibility | Arms on one GRU init and 1e7 real-text bytes: random features (readout only), local rule (e-prop + feedback alignment on embedding/core), matched perturbation, online-BPTT CONTROL; representation probe ΔR, headroom ρ, novel-context BPB, retention κ | **Gate A (decision):** validity V1-V4 (else INCONCLUSIVE); PASS needs P1-P5 with thresholds frozen from pilots in `model/specs/representation.json`; FAIL / PIVOT needs a written decision in `research/decisions/` before Phase 10 |
| 10 Scalable core + seed circuits + selected rule | Bake-off at scale, circuits C1-C4, C7, Gate A's rule on the ISA, unit embeddings, consolidation | **Gate B / L1:** BPB <= PPM/KN-5 at 1e8 bytes; learned blocks still contribute; >= 1,650 bytes/s |
| 11 Scaling + forgetting | 1e8-1e9 bytes, 2-3 sizes, scaling curves | **Gate C / L2**; stop on plateau |
| 12 Read into memory + reasoning | Learned write gate, learned text-to-bindings, C5/C6 driven by network | Read-then-answer beats LM alone and lookup; ablation removes gain |
| 13 Generation + dialogue | Decoding, human-written dialogue exposure, abstention | Non-template generation, blind ratings, calibrated |
| 14 Self-proposed WSP | Revision-1 Phase 9 on the Phase 7 compiler | Useful proposed patch passes hidden eval, no regressions |
| 15 Evaluation + levels | Revision-1 Phase 10 framed by L1-L4 | Published level profile, 3 seeds, hidden sets |
| 16 Release 1.0 | Revision-1 Phase 11 | Requires L3 |

Track G (grounded world, revision-1 Phase 7) is optional and parallel from Phase 10. Revision-1 phase texts are in ROADMAP Appendix A.
