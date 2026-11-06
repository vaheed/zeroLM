# ZeroLM

> **Zero offline pretraining. Lifelong online learning. No imported model weights.**

ZeroLM is a research project to build a CPU-first neural language system in Rust that starts with no pretrained weights and develops its parameters and memory only from its own chronological stream of input, prediction, feedback, and interaction.

It is **not** a small Transformer with the KV cache removed. The target is a recurrent, online-learning system with fast associative weights, persistent weight-resident memory, and a controlled weight-space programming layer.

## Quick start

```bash
docker compose up --build                      # guided demo: reasoning, memory, online learning, internals shown
docker compose run --rm reason                 # talk to the reasoner yourself
docker compose run --rm gates                  # run every gate and get a pass/fail table
docker compose -f docker-compose.ghcr.yml up   # same demo from the prebuilt image ghcr.io/vaheed/zerolm
```

Add `--verbose` to `zerolm run`, `memory`, or `reason` (on by default in Docker) to see each step: the parse, the tensors written, the operators and premises behind every answer, and the checkpoint hash. Full guide with examples and how to check each claim: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## Status

**Phases 0-6 are done: research constitution, provenance baseline, a CPU neural engine (GRU core), a byte-level bootstrap with a teaching protocol and CLI, a first online learner, weight-resident memory, and compositional reasoning over bound vectors.** A local delta rule on the readout gets held-out text from 8.0 to about 2.3 bits/byte after one chronological pass (byte-frequency baseline: 4.2; a byte trigram is still better at 1.2). See [docs/ONLINE_LEARNING.md](docs/ONLINE_LEARNING.md). Taught facts now persist across restarts only as checkpoint tensors and are answered by an attention read (1,200 facts per seed: exact recall 1.000, paraphrase recall 0.973). See [docs/MEMORY_FORMAT.md](docs/MEMORY_FORMAT.md). Statements of a bounded language are stored as role-filler vectors, and eight programmed operators answer new compositional questions from them (100% agreement with a symbolic reference on 3 seeds plus a hidden seed; a lookup baseline gets 0% of derived questions). See [docs/REASONING.md](docs/REASONING.md). No claim of English understanding is made.

| Phase | Topic | Status |
|---|---|---|
| 0 | Research constitution and scope freeze | complete |
| 1 | Freeze and audit the baseline | complete |
| 2 | Neural compute engine and architecture bake-off | complete |
| 3 | Byte bootstrap and deterministic initialization | complete |
| 4 | First true online language model (**stop/go gate**) | complete (go, weak learner) |
| 5 | Weight-resident associative memory | complete |
| 6 | Concepts, relations, composition | complete |
| 7 | Uniform neural engine and WSP compiler | next |
| 8 | Real-text exposure streams and reference baselines | planned |
| 9 | Scalable core with compiled seed circuits (Gate A) | planned |
| 10 | Deep representation learning without backpropagation (Gate B) | planned |
| 11 | Scaling run and forgetting (Gate C) | planned |
| 12 | Reading into memory and reasoning inside the network | planned |
| 13 | Generation, instruction following, and dialogue | planned |
| 14 | Self-proposed WSP patches | planned |
| 15 | Capability evaluation and level certification | planned |
| 16 | CPU deployment and 1.0 release (requires Level L3, a small LLM) | planned |

Phases 7-16 follow roadmap revision 2 ([docs/ROADMAP.md](docs/ROADMAP.md) sec. 0.3): one neural engine whose behavior lives in weights, weight-space programming as a compiler of algorithms (never language data), and online learning from licensed human-written text toward a small but genuine LLM.

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

## Try it with Cargo

```bash
cargo test --workspace
cargo run -p zerolm-policy --bin zerolm-audit          # Phase 0 gate
cargo run -p zerolm-policy --bin zerolm-audit -- --json
cargo run -p zerolm-policy --bin zerolm-baseline -- generate   # then: check
cargo run --release -p zerolm-eval --bin zerolm-bakeoff -- --check model/specs/architecture.json   # Phase 2 gate
cargo run -p zerolm-runtime -- info     # Phase 3: pinned init hash
printf 'teach sky => blue\nask sky\n' | cargo run -p zerolm-runtime -- run
cargo run --release -p zerolm-eval --bin zerolm-online-lm   # Phase 4 gate (3 seeds)
cargo run --release -p zerolm-eval --bin zerolm-memory      # Phase 5 gate (3 seeds x 1,200 facts)
printf 'teach sky color => blue\n' | cargo run -p zerolm-runtime -- memory --dir /tmp/zlm
printf 'ask what is the color of the sky?\n' | cargo run -p zerolm-runtime -- memory --dir /tmp/zlm   # new process
cargo run --release -p zerolm-eval --bin zerolm-reasoning   # Phase 6 gate (3 seeds + hidden seed)
printf 'every dog is a mammal\nrex is a dog\nis rex a mammal?\n' | cargo run -p zerolm-runtime -- reason
cargo run -p zerolm-runtime -- reason --verbose < examples/reasoning-more.txt   # see inside each step
```

The audit checks the update policy and seed-knowledge manifests, that every weight group declares an init source and update path, that no phase is marked complete without a command and schema, and that the tree contains no pretrained weight files or model downloads.

## Repository layout

```text
crates/zerolm-policy/     Phase 0-1 auditor and baseline tools
crates/zerolm-tensor/     f32 kernels, deterministic RNG and init
crates/zerolm-nn/         GRU, SSM, fast-weight, hybrid cells
crates/zerolm-tokenizer/  byte tokenizer
crates/zerolm-learning/   online predictor, local rules, streams, n-gram baselines
crates/zerolm-memory/     weight-resident memory, retrieval, W_FAST, atomic checkpoints
crates/zerolm-reasoner/   relational vector memory, inference operators, symbolic oracle
crates/zerolm-engine/     model forward pass, exact state resume, binding, semantic parser
crates/zerolm-eval/       architecture bake-off, online-LM, memory, and reasoning benchmarks
crates/zerolm-runtime/    `zerolm` CLI (info, run, memory, reason)
examples/                 input files for the CLI and the demo
scripts/docker/           demo and gate scripts used by the Docker image
Dockerfile, docker-compose*.yml   build-locally and prebuilt-image ways to run
research/manifest/        seed-knowledge.json, update-policy.json
research/phases.json      phase status and gates
model/specs/              weight-group declarations
docs/                     roadmap, constitution, claims and limitations
AGENTS.md                 rules and workflow for contributors and AI agents
.claude/skills/           roadmap skill generated from docs/ROADMAP.md
```

## Documents

- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md): run, watch internals, examples, check every claim
- [docs/ROADMAP.md](docs/ROADMAP.md): full research and engineering plan, risks, kill criteria
- [docs/RESEARCH_CONSTITUTION.md](docs/RESEARCH_CONSTITUTION.md): definitions and invariants
- [docs/CLAIMS_AND_LIMITATIONS.md](docs/CLAIMS_AND_LIMITATIONS.md): what is and is not claimed
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): engine, candidates, bake-off results and caveats
- [docs/BOOTSTRAP.md](docs/BOOTSTRAP.md): byte interface, teaching protocol, what is not claimed
- [docs/BASELINE.md](docs/BASELINE.md): baseline and reproduction
- [AGENTS.md](AGENTS.md): working rules

## License

Apache-2.0. See [LICENSE](LICENSE).
