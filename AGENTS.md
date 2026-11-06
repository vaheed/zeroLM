# AGENTS.md

ZeroLM: a CPU-first, Rust, online-learning neural language system with **zero offline pretraining**. Full plan: `docs/ROADMAP.md`. Rules: `docs/RESEARCH_CONSTITUTION.md`.

## Layout
- `crates/` - Rust workspace. Today: `zerolm-policy` (audit + baseline tools), `zerolm-tensor`, `zerolm-nn`, `zerolm-tokenizer`, `zerolm-engine`, `zerolm-eval`. Planned: learning, memory, reasoner, world, wsp, runtime (roadmap sec. 9). Selected core: GRU (`model/specs/architecture.json`); the bake-off check fails on selection drift.
- `research/manifest/` - `seed-knowledge.json`, `update-policy.json` (the contract).
- `research/phases.json` - phase status; `complete` requires `command` + `report_schema`.
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
```
All of these must pass before every commit.

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
