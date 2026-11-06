# ZeroLM

A CPU-first neural language system in Rust with **zero offline pretraining**: parameters and memory develop only from their own chronological stream of input and feedback. Online learning is allowed and reported honestly.

**Status:** Phase 0 (research constitution) only. No language model exists yet. See [docs/ROADMAP.md](docs/ROADMAP.md), [docs/RESEARCH_CONSTITUTION.md](docs/RESEARCH_CONSTITUTION.md), and [docs/CLAIMS_AND_LIMITATIONS.md](docs/CLAIMS_AND_LIMITATIONS.md).

```
cargo test --workspace
cargo run -p zerolm-policy --bin zerolm-audit     # Phase 0 gate
```

Contributors and agents: read [AGENTS.md](AGENTS.md).
