# ZeroLM

> **Zero offline pretraining. Lifelong online learning. No imported model weights.**

ZeroLM is a research project to build a CPU-first neural language system in Rust that starts with no pretrained weights and develops its parameters and memory only from its own chronological stream of input, prediction, feedback, and interaction.

It is **not** a small Transformer with the KV cache removed. The target is a recurrent, online-learning system with fast associative weights, persistent weight-resident memory, and a controlled weight-space programming layer.

## Status

**Phases 0-4 are done: research constitution, provenance baseline, a CPU neural engine (GRU core), a byte-level bootstrap with a teaching protocol and CLI, and a first online learner.** A local delta rule on the readout gets held-out text from 8.0 to about 2.3 bits/byte after one chronological pass (byte-frequency baseline: 4.2; a byte trigram is still better at 1.2). See [docs/ONLINE_LEARNING.md](docs/ONLINE_LEARNING.md). No claim of English understanding is made.

| Phase | Topic | Status |
|---|---|---|
| 0 | Research constitution and scope freeze | complete |
| 1 | Freeze and audit the baseline | complete |
| 2 | Neural compute engine and architecture bake-off | complete |
| 3 | Byte bootstrap and deterministic initialization | complete |
| 4 | First true online language model (**stop/go gate**) | complete (go, weak learner) |
| 5 | Weight-resident associative memory | planned |
| 6 | Concepts, relations, composition | planned |
| 7 | Grounded learning and interaction | planned |
| 8 | Generation and multi-turn dialogue | planned |
| 9 | Controlled self-modification (WSP) | planned |
| 10 | Developmental capability evaluation | planned |
| 11 | CPU deployment and 1.0 release | planned |

Status is machine-readable in [research/phases.json](research/phases.json). A phase counts as complete only with an executable command, a report schema, a CI job, and uploaded evidence.

## Target architecture

```text
UTF-8 bytes / feedback / environment
        -> byte-level interface (no tokenizer, no vocabulary)
        -> recurrent core (gated linear recurrence; GRU and local-attention as baselines)
              |-> next-byte distribution
              |-> fast plastic weights   (delta-rule updates, online)
              |-> persistent memory      (key-query tensors in the checkpoint)
        -> language decoder
        -> online learning controller
              |-> local plasticity updates
              |-> WSP patch proposals -> validator + regression gate -> accept / reject / rollback
```

Weight groups (declared in [model/specs/weight-groups.json](model/specs/weight-groups.json)): core, embeddings, fast weights, memory, skills, meta. Slow groups change only through validated patches; fast groups and memory change through explicit local rules.

## Principles

- **Learning is allowed, and reported honestly.** Every update records its input stream, signal, rule, changed tensors, and checkpoint hash.
- **No online backpropagation in the mainline.** Local plasticity and explicit weight edits come first; a gradient run is allowed only as a labelled control.
- **The model is never its own teacher.** Updates come from the observed next byte, a user correction, an environment outcome, or an independently validated signal.
- **Strict data roles.** Exposure, development, and hidden evaluation streams stay separate; hidden labels never reach the update controller.
- **Every patch is attributable, reversible, and regression-tested.**
- **Falsifiable steps.** If Phase 4 shows no held-out acquisition from chronological input, the project stops claiming a self-developing language model.

## What this does not promise

No seven-year-old or twenty-year-old cognition, and no age-equivalent claims. Those are aspiration labels; results are reported as concrete capabilities on named benchmarks. See [docs/CLAIMS_AND_LIMITATIONS.md](docs/CLAIMS_AND_LIMITATIONS.md).

## Try it

```bash
cargo test --workspace
cargo run -p zerolm-policy --bin zerolm-audit          # Phase 0 gate
cargo run -p zerolm-policy --bin zerolm-audit -- --json
cargo run -p zerolm-policy --bin zerolm-baseline -- generate   # then: check
cargo run --release -p zerolm-eval --bin zerolm-bakeoff -- --check model/specs/architecture.json   # Phase 2 gate
cargo run -p zerolm-runtime -- info     # Phase 3: pinned init hash
printf 'teach sky => blue\nask sky\n' | cargo run -p zerolm-runtime -- run
cargo run --release -p zerolm-eval --bin zerolm-online-lm   # Phase 4 gate (3 seeds)
```

The audit checks the update policy and seed-knowledge manifests, that every weight group declares an init source and update path, that no phase is marked complete without a command and schema, and that the tree contains no pretrained weight files or model downloads.

## Repository layout

```text
crates/zerolm-policy/     Phase 0-1 auditor and baseline tools
crates/zerolm-tensor/     f32 kernels, deterministic RNG and init
crates/zerolm-nn/         GRU, SSM, fast-weight, hybrid cells
crates/zerolm-tokenizer/  byte tokenizer
crates/zerolm-learning/   online predictor, local rules, streams, n-gram baselines
crates/zerolm-engine/     model forward pass, exact state resume
crates/zerolm-eval/       architecture bake-off
crates/zerolm-runtime/    `zerolm` CLI (info, run)
research/manifest/        seed-knowledge.json, update-policy.json
research/phases.json      phase status and gates
model/specs/              weight-group declarations
docs/                     roadmap, constitution, claims and limitations
AGENTS.md                 rules and workflow for contributors and AI agents
.claude/skills/           roadmap skill generated from docs/ROADMAP.md
```

## Documents

- [docs/ROADMAP.md](docs/ROADMAP.md): full research and engineering plan, risks, kill criteria
- [docs/RESEARCH_CONSTITUTION.md](docs/RESEARCH_CONSTITUTION.md): definitions and invariants
- [docs/CLAIMS_AND_LIMITATIONS.md](docs/CLAIMS_AND_LIMITATIONS.md): what is and is not claimed
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): engine, candidates, bake-off results and caveats
- [docs/BOOTSTRAP.md](docs/BOOTSTRAP.md): byte interface, teaching protocol, what is not claimed
- [docs/BASELINE.md](docs/BASELINE.md): baseline and reproduction
- [AGENTS.md](AGENTS.md): working rules

## License

Apache-2.0. See [LICENSE](LICENSE).
