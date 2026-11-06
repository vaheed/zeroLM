# AGENTS.md

ZeroLM: a CPU-first, Rust, online-learning neural language system with **zero offline pretraining**. Full plan: `docs/ROADMAP.md`. Rules: `docs/RESEARCH_CONSTITUTION.md`.

## Layout
- `crates/` - Rust workspace. Today: `zerolm-policy` (audit + baseline tools), `zerolm-tensor`, `zerolm-nn`, `zerolm-tokenizer`, `zerolm-engine` (incl. `bootstrap`), `zerolm-learning` (online predictor), `zerolm-memory` (weight-resident memory, checkpoint format), `zerolm-reasoner` (relational vector memory, operators, oracle), `zerolm-eval` (bake-off, online-LM, memory, reasoning benchmarks; `isa_ref` reference interpreter), `zerolm-runtime` (CLI `zerolm`), `zerolm-wsp` (circuit compiler, patches; Phase 7). Planned (roadmap revision 2): `zerolm-data` (corpus manifest, streams); `zerolm-world` only for optional Track G. From Phase 7 on, model behavior must live in weights run by the neural ISA (instruction set architecture: the fixed tensor-op set; `zerolm-engine/src/isa/`, `docs/NEURAL_ISA.md`; constitution invariants 10-13). Circuit specs live in `model/prog/`; their compiled hashes are pinned in `model/prog/pinned.json` (re-pin only for a deliberate change; `MemoryConfig::default()` must compile to the shipped `memory-read` spec). Phase 5 queries run only through the compiled `memory-read` circuit. `model/bootstrap/seed.json` pins the init hash; changing initialization requires deliberately re-pinning it. The protocol ledger is scaffolding, never evidence of memory or learning. Selected core: GRU (`model/specs/architecture.json`); the bake-off check fails on selection drift. Phase 4 rule constants, seeds, and the improvement floor are frozen in `model/specs/online-learning.json` (tune only on pilot seeds). Phase 5 memory config and gates are frozen in `model/specs/memory.json`, and `MemoryConfig::default()` must equal it; facts live only in checkpoint tensors (`docs/MEMORY_FORMAT.md`), never in side files. Phase 6 config and gates are frozen in `model/specs/reasoning.json` (default config must equal it); the symbolic oracle and lookup baseline are evaluation-only and must never be called by the reasoner (`docs/REASONING.md`).
- `research/manifest/` - `seed-knowledge.json`, `update-policy.json` (the contract).
- `research/streams/` - generated exposure/held-out/novel stream specs.
- `research/phases.json` - phase status (phases 7-16 per roadmap revision 2); `complete` requires `command` + `report_schema`.
- `model/specs/weight-groups.json` - every weight group's `init_source` and `update_path`.
- `reports/<phase>/` - generated evidence (git-ignored).
- No legacy code exists; there is no backward-compatibility requirement.

## Commands
```
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p zerolm-policy --bin zerolm-audit -- --json   # Phase 0 gate
cargo run -p zerolm-policy --bin zerolm-baseline -- generate && cargo run -p zerolm-policy --bin zerolm-baseline -- check   # Phase 1 gate (needs a clean tree)
cargo run --release -p zerolm-eval --bin zerolm-bakeoff -- --check model/specs/architecture.json   # Phase 2 gate (~2 min)
cargo run --release -p zerolm-eval --bin zerolm-online-lm   # Phase 4 gate (~15 s)
cargo run --release -p zerolm-eval --bin zerolm-memory      # Phase 5 gate, incl. reference-interpreter parity (~7 min)
cargo run --release -p zerolm-eval --bin zerolm-reasoning   # Phase 6 gate (~90 s)
cargo run -p zerolm-wsp -- check                            # Phase 7: compiled circuits match model/prog/pinned.json
```
All of these must pass before every commit. In Docker: `docker compose run --rm gates` (all but Phase 1, which needs a clean git checkout); guide in `docs/GETTING_STARTED.md`.

## Non-negotiable rules
1. No pretrained weights, tokenizers, or model downloads. No files with extensions listed in `update-policy.json`.
2. No online backpropagation in the mainline.
3. Never train on hidden evaluation data; never use the model's own output as a teacher.
4. Every update logs input stream, signal, rule, changed tensors, checkpoint hash.
5. No arbitrary-code execution from self-modification.
6. New weight groups must be declared in `weight-groups.json` before use.
7. Never claim learning from a changed weight file; require a before/after behavioral metric.
8. A phase is `complete` only with an executable command, report schema, CI job, and uploaded evidence. A CI job must never silently skip because a script is missing.

## Workflow
- One phase at a time, in roadmap order; small commits, one per task, each with tests.
- Commit as the repository owner only; no AI attribution lines.
- Update `docs/CLAIMS_AND_LIMITATIONS.md` and `research/phases.json` when a phase lands.
- Rust: stable 1.75+, no `unsafe` without justification, minimal dependencies, deterministic seeded randomness.
- Skill: `.claude/skills/zerolm-roadmap/SKILL.md` summarizes the phases and gates.
