# Claims and Limitations

## What is claimed (Phases 0-2)
- The repository defines, and CI enforces, what is hand-authored, what may be learned online, and what is prohibited.
- A CPU engine exists: deterministic kernels, four recurrent cells at equal parameter budgets, fixed-size exactly-resumable session state, measured at initialization. The GRU core was selected by a documented, reproducible score.
- Baseline reports record commit, clean-tree status, and toolchain; `zerolm-baseline check` rejects dirty or missing provenance.

## What is not claimed
- No language model exists yet: the engine has no learning rule, so it predicts at chance (~8 bits/byte). Phases 3-11 are unimplemented.
- Bake-off metrics are initialization-time capacity measurements, not language ability, memory capacity, or learning evidence.
- No capability, benchmark score, or "human-level" claim is made.
- Memory is finite; capacity and interference will be measured, not assumed.
- The static scan detects obvious pretrained artifacts and download calls; it cannot prove the absence of every hidden data source.
- Code and data from the removed v0.2 / ZeroLM Next experiments live only in git history; no results from them carry over.

Update this file whenever a phase lands.
