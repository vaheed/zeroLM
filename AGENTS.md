# AGENTS.md

ZeroLM: a CPU-first, Rust, online-learning neural language system with **zero offline pretraining**. Full plan: `docs/roadmap.md`. Rules: `docs/research-constitution.md`.

## Layout
- `crates/` - Rust workspace. Today: `zerolm-policy` (audit + baseline tools), `zerolm-tensor`, `zerolm-nn`, `zerolm-tokenizer`, `zerolm-engine` (incl. `bootstrap`), `zerolm-learning` (online predictor), `zerolm-memory` (weight-resident memory, checkpoint format), `zerolm-reasoner` (relational vector memory, operators, oracle), `zerolm-eval` (bake-off, online-LM, memory, reasoning benchmarks; `isa_ref` reference interpreter), `zerolm-runtime` (CLI `zerolm`), `zerolm-wsp` (circuit compiler, patches). Planned (roadmap revision 2): `zerolm-data` (corpus manifest, streams); `zerolm-world` only for optional Track G. From the uniform-engine milestone on, model behavior must live in weights run by the neural ISA (instruction set architecture: the fixed tensor-op set; `zerolm-engine/src/isa/`, `docs/neural-isa.md`; constitution invariants 10-13). Circuit specs live in `model/prog/`; their compiled hashes are pinned in `model/prog/pinned.json` (re-pin only for a deliberate change; `MemoryConfig::default()` must compile to the shipped `memory-read` spec). Memory queries run only through the compiled `memory-read` circuit. `model/bootstrap/seed.json` pins the init hash; changing initialization requires deliberately re-pinning it. The protocol ledger is scaffolding, never evidence of memory or learning. Selected core: GRU (`model/specs/architecture.json`); the bake-off check fails on selection drift. Online-learning rule constants, seeds, and the improvement floor are frozen in `model/specs/online-learning.json` (tune only on pilot seeds). Memory config and gates are frozen in `model/specs/memory.json`, and `MemoryConfig::default()` must equal it; facts live only in checkpoint tensors (`docs/memory-format.md`), never in side files. Reasoning config and gates are frozen in `model/specs/reasoning.json` (default config must equal it); the symbolic oracle and lookup baseline are evaluation-only and must never be called by the reasoner (`docs/reasoning.md`).
- `research/manifest/` - `seed-knowledge.json`, `update-policy.json` (the contract).
- `research/streams/` - generated exposure/held-out/novel stream specs.
- `research/milestones.json` - roadmap milestones by name, in roadmap order, with status; `complete` requires `command` + `report_schema`. Phase numbers appear only in `docs/roadmap.md`; code, CI, reports, and other docs use milestone names.
- `model/specs/weight-groups.json` - every weight group's `init_source` and `update_path`.
- `reports/<name>/` - generated evidence (git-ignored), e.g. `reports/memory/`.
- No legacy code exists; there is no backward-compatibility requirement.

## Commands
```
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p zerolm-policy --bin zerolm-audit -- --json   # policy audit
cargo run -p zerolm-policy --bin zerolm-baseline -- generate && cargo run -p zerolm-policy --bin zerolm-baseline -- check   # baseline provenance (needs a clean tree)
cargo run --release -p zerolm-eval --bin zerolm-bakeoff -- --check model/specs/architecture.json   # bake-off gate (~2 min)
cargo run --release -p zerolm-eval --bin zerolm-online-lm   # online-learning gate (~15 s)
cargo run --release -p zerolm-eval --bin zerolm-memory      # memory gate, incl. reference-interpreter parity (~7 min)
cargo run --release -p zerolm-eval --bin zerolm-reasoning   # reasoning gate (~2 min)
cargo run -p zerolm-wsp -- check                            # compiled circuits match model/prog/pinned.json
cargo run --release -p zerolm-eval --bin zerolm-uniform    # uniform-engine gate: reference replay of every circuit run, host audit (~2 min)
```
All of these must pass before every commit. In Docker: `docker compose run --rm gates` (all but baseline provenance, which needs a clean git checkout); guide in `docs/getting-started.md`.

## Non-negotiable rules
1. No pretrained weights, tokenizers, or model downloads. No files with extensions listed in `update-policy.json`.
2. No online backpropagation in the mainline.
3. Never train on hidden evaluation data; never use the model's own output as a teacher.
4. Every update logs input stream, signal, rule, changed tensors, checkpoint hash.
5. No arbitrary-code execution from self-modification.
6. New weight groups must be declared in `weight-groups.json` before use.
7. Never claim learning from a changed weight file; require a before/after behavioral metric.
8. A milestone is `complete` only with an executable command, report schema, CI job, and uploaded evidence. A CI job must never silently skip because a script is missing.

## Workflow
- One milestone at a time, in roadmap order; small commits, one per task, each with tests.
- Commit as the repository owner only; no AI attribution lines.
- Update `docs/claims-and-limitations.md` and `research/milestones.json` when a milestone lands.
- Rust: stable 1.75+, no `unsafe` without justification, minimal dependencies, deterministic seeded randomness.
- Skill: `.claude/skills/zerolm-roadmap/SKILL.md` summarizes the roadmap and its gates.
