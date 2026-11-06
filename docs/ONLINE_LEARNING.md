# Online learning (Phase 4)

Phase 4 asks whether the same untrained architecture gets better at predicting **unseen** text after a chronological stream of observations, using local update rules and no offline training loop.

Short answer: **yes, measurably, but only in a narrow sense.** A local rule on the readout layer brings held-out bits-per-byte from 8.0 (untrained) down to about 2.3, and beats the byte-frequency baseline by about 1.9 BPB on sentences and word combinations never seen in exposure. A plain byte trigram is still much better (about 1.2 BPB). This is a first, weak, honest language learner, not a competitive one.

## What learns and what does not

```text
byte -> embedding (frozen, seeded) -> GRU core (frozen, seeded) -> h
features phi = [h ; embedding(last byte)]                          (90 values)
p(next byte) = softmax(readout.weight * phi + readout.bias)       <- the only plastic tensors
```

| Tensor group | Init | Updated in Phase 4 |
|---|---|---|
| `byte_embed` | seeded PRNG | no |
| `recurrent_core` (GRU) | seeded PRNG | no |
| `readout_head` (`readout.weight` 256x90, `readout.bias` 256) | zeros (uniform prediction) | yes, by the configured rule |

The Phase 3 `Model.head` stays in the weight hash but is not used by the online predictor. Its job goes to the zero-initialized readout.

## Protocol per byte

1. Predict a distribution from the features of the bytes consumed so far.
2. Reveal the actual byte. The only signal is `observed_next_byte`.
3. If learning is enabled, apply the rule to the readout.
4. Log the event, then feed the byte into the recurrent state.

`OnlinePredictor::evaluate` takes `&self`, so the type system stops evaluation data from reaching an update. A predictor built with `Rule::Frozen`, or with `set_learning(false)`, logs no events and keeps its weight hash.

## Rules

| Rule | Update for target `y`, prediction `p`, features `phi` | Error signal? |
|---|---|---|
| `frozen` | none | none |
| `delta` (primary) | `dW = lr (onehot(y) - p) phi^T / (|phi|^2 + eps)`, bounded Frobenius norm; `db = lr_b (onehot(y) - p)` | yes |
| `hebbian` | decay all weights by `1 - decay`, then `W[y] += lr * phi/|phi|`, `b[y] += lr`; predict with `gain * cos(W_i, phi) + ln(b_i + prior)` | no |
| `delta_no_recurrence` | `delta` with the GRU's recurrent path severed (ablation) | yes |

The constants are in [model/specs/online-learning.json](../model/specs/online-learning.json) and registered as the `delta-rule-constants` seed entry. They were chosen on pilot seeds 9001-9002 only, then frozen with the improvement floor before any main-seed run.

**Limitation stated plainly.** On a softmax readout, the normalized delta rule is the same as one cross-entropy gradient step **for that single layer**. It is local (the outer product of the output error and the input activity), with no backpropagation through layers or time. It is not "gradient-free" in a strict sense. The Hebbian rule uses no prediction error. The online truncated-BPTT control from the roadmap was **not run**: the project has no backpropagation code, and the gate was decided without it.

## Streams and data roles

[research/streams/compositional-v1.json](../research/streams/compositional-v1.json) is a small hand-written lexicon with 6 sentence templates. It is exposure data, and nothing initializes weights from it. For each seed:

- **exposure**: 40,000 bytes, observed once in order. This is training input.
- **heldout**: sentences whose exact text never occurs in exposure.
- **novel**: sentences that bind at least one of 12 held-out slot combinations (for example "green kite" or "zoe drops"). Exposure never contains these combinations.

Contamination is counted, not assumed. The run fails if any held-out or novel sentence appears in exposure, or if an exposure sentence contains every word of a held-out combination.

## Logging

Every update event records `seq`, `stream_offset`, `target`, `predicted`, `p_target`, and `update_norm`. Every 1,000 events, a checkpoint record stores the policy fields (`input_stream`, `signal`, `rule`, `changed_tensors`, `checkpoint_sha256`) plus `targets_sha256` over that block's target bytes. Full weights are hashed at block boundaries rather than per byte, because hashing every byte would cost more than the update itself. The benchmark checks that the events match the stream byte for byte, that the blocks are contiguous and cover the stream, that the block target hashes match, and that the last checkpoint hash equals the final weights.

## Results (main seeds 1, 2, 3; 40,000 exposure events each)

Reproduce with `cargo run --release -p zerolm-eval --bin zerolm-online-lm` (about 15 s on 4+ cores). Reports go to `reports/online-learning/`.

| Arm | Held-out BPB | Novel-combination BPB | Held-out accuracy |
|---|---|---|---|
| frozen (same architecture, untrained) | 8.000 | 8.000 | 0.000 |
| **delta** | **2.282 ± 0.084** | **2.356 ± 0.051** | 0.485 |
| hebbian | 3.293 ± 0.056 | 3.331 ± 0.049 | 0.354 |
| delta, recurrence severed | 2.912 ± 0.063 | 2.973 ± 0.092 | 0.370 |
| byte-frequency baseline | 4.200 ± 0.016 | 4.202 ± 0.014 | |
| byte trigram baseline | 1.209 ± 0.024 | 1.214 ± 0.015 | |
| byte 5-gram baseline | 0.661 ± 0.023 | 0.857 ± 0.032 | |

The gate compares each rule with min(frozen, byte-frequency). The floor is 0.25 BPB, fixed before the main run. Every seed must clear it, and so must mean − 2 sd.

| Rule | Held-out improvement per seed | Novel improvement per seed | Passed |
|---|---|---|---|
| delta | 2.011, 1.847, 1.896 | 1.917, 1.804, 1.817 | yes |
| hebbian | 0.968, 0.859, 0.893 | 0.929, 0.817, 0.865 | yes |

Additional evidence:
- **Not only memorization:** on novel combinations, the delta rule is only 0.23 BPB worse than when it re-reads its own exposure. Novel-combination BPB is within 0.08 of held-out BPB.
- **The recurrent state is used:** severing the GRU recurrence costs the delta rule 0.63 held-out BPB.
- Learning curves (BPB, accuracy, ECE at 0, 250, ..., 40,000 events) are in `learning-curve.csv`.

## What this does not show

- No English understanding. The text is a tiny generated language with about 50 words.
- No competitive language modeling. A byte trigram is about 1 BPB better. Only 23k readout parameters learn, on top of frozen random features.
- No memory beyond the GRU state's short horizon, and no facts stored across sessions (that is Phase 5).
- Forgetting and longer streams have not been measured yet. Learning saturates within about 20k events at this capacity.
