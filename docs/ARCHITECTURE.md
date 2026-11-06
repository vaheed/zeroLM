# Architecture (Phase 2)

## Data flow implemented so far
```text
UTF-8 bytes -> tokenizer (ids + offsets) -> embedding (256 x 16) -> recurrent cell -> next-byte head
```
Session state is one fixed-size `f32` vector per cell. Nothing grows with stream length; there is no KV cache. State serializes exactly (header, f32 bits, checksum) and resumes bit-identically.

## Crates
| Crate | Contents |
|---|---|
| `zerolm-tensor` | Matrix and vector kernels, SplitMix64 RNG, tagged deterministic init, finite checks. Plain loops, no `unsafe`, no SIMD/BLAS yet. |
| `zerolm-nn` | `Cell` trait, four cells, trace hooks, ablation hooks, state serialization. No backpropagation exists. |
| `zerolm-tokenizer` | Bytes 0..255 as IDs, control IDs BOS/EOS/PAD, offsets. No vocabulary. |
| `zerolm-engine` | `Model`: embed, cell, head; `feed`, `bits_per_byte`, save/load state. |
| `zerolm-learning` | Phase 4 online predictor (plastic readout over frozen GRU features), local rules, generated streams, byte n-gram baselines. See `docs/ONLINE_LEARNING.md`. |
| `zerolm-eval` | Bake-off metrics and the `zerolm-bakeoff` binary; the Phase 4 `zerolm-online-lm` benchmark. |

## Candidates (equal parameter budget, within 5%)
1. `gru`: compact GRU, update-gate biases spread over timescales.
2. `ssm`: diagonal gated linear recurrence with input-dependent decay. Its chunked scan matches token-by-token stepping to 1e-5.
3. `fast_weight`: leaky recurrent state plus an associative matrix M updated by the normalized delta rule (`decay 0.995`, `lr 0.5`, clamp 8). M is session state.
4. `hybrid`: GRU plus attention over a fixed window of 8 past states. Comparison arm: never selectable.

## Selection
<a name="selection"></a>
Score (deterministic; latency and RSS are reported and gated, but not scored because wall-clock is noisy):
`0.60 * log-scaled retention horizon + 0.25 * compactness + 0.15 * traceability`, over stable, eligible candidates, with a hard p95 latency gate of 2 ms per token.

Latest full run (budget 20,000 cell params, seeds 1-3, release build, one 32-core Xeon E5-2697 v4):

| candidate | hidden | state bytes | horizon (lags) | p50 us | p95 us | score |
|---|---|---|---|---|---|---|
| **gru** (selected) | 74 | 296 | 128 | 38.5 | 94.6 | 0.883 |
| ssm | 606 | 2,424 | 64 | 210.0 | 459.5 | 0.738 |
| fast_weight | 88 | 15,840 | 128 | 53.0 | 75.9 | 0.600 |
| hybrid (ineligible) | 58 | 2,096 | 256 | 40.2 | 71.9 | 0.967 |

Reproduce: `cargo run --release -p zerolm-eval --bin zerolm-bakeoff -- --check model/specs/architecture.json`. The check fails if `model/specs/architecture.json` disagrees with the measured selection.

## What this does and does not show
- Metrics are taken **at seeded initialization; nothing is trained or learned**. The retention horizon is the largest lag at which the hidden state still differs measurably (relative difference >= 1e-3) when only the first byte changes. It shows how long information *can* persist, not that a later learner can use it.
- GRU and fast_weight tie on horizon; GRU wins on compactness (a 296-byte state vs 15.8 KB, because M is state). The choice is sensitive to the weights in the score. With horizon alone the hybrid would lead, but it is a comparison arm by the roadmap.
- The fast-weight memory is **not evaluated as a memory** here (capacity, interference, recall); that is Phase 5. Selecting the GRU as the Phase 3-4 core does not rule out adding the delta-rule memory later.
- Chronological learning, held-out generalization, and forgetting cannot be measured until Phase 4.
- Bits per byte of the untrained models is about 8.0, as expected for random weights.
- The independent kernel reference is a naive `f64` implementation in Rust tests, not Python (the project has no Python).
- Linux was run locally; macOS builds are exercised by the CI matrix only.
