# Architecture (Phase 2)

## Data flow implemented so far
```text
UTF-8 bytes -> tokenizer (ids + offsets) -> embedding (256 x 16) -> recurrent cell -> next-byte head
```
Session state is one fixed-size `f32` vector per cell. Nothing grows with stream length; there is no KV cache (key-value cache: the per-token store of past keys and values that a Transformer keeps and that grows with the context). State serializes exactly (header, f32 bits, checksum) and resumes bit-identically.

## Crates
| Crate | Contents |
|---|---|
| `zerolm-tensor` | Matrix and vector kernels, SplitMix64 RNG, tagged deterministic init, finite checks. Plain loops, no `unsafe`, no SIMD/BLAS yet. |
| `zerolm-nn` | `Cell` trait, four cells, trace hooks, ablation hooks, state serialization. No backpropagation exists. |
| `zerolm-tokenizer` | Bytes 0..255 as IDs, control IDs BOS/EOS/PAD, offsets. No vocabulary. |
| `zerolm-engine` | `Model`: embed, cell, head; `feed`, `bits_per_byte`, save/load state. `binding.rs` (phasor role-filler binding) and `semantic_state.rs` (bounded parser, discourse state) for Phase 6. |
| `zerolm-learning` | Phase 4 online predictor (plastic readout over frozen GRU features), local rules, generated streams, byte n-gram baselines. See `docs/ONLINE_LEARNING.md`. |
| `zerolm-memory` | Phase 5 weight-resident memory: semantic slots, episodic ring, W_FAST, attention retrieval, atomic named-tensor checkpoints. See `docs/MEMORY_FORMAT.md`. |
| `zerolm-reasoner` | Phase 6 relational memory of bound role-filler vectors, eight inference operators, the symbolic oracle and lookup baseline (evaluation only). See `docs/REASONING.md`. |
| `zerolm-eval` | Bake-off metrics and the `zerolm-bakeoff` binary; the Phase 4 `zerolm-online-lm`, Phase 5 `zerolm-memory`, and Phase 6 `zerolm-reasoning` benchmarks. |

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

## Mathematics

Notation and abbreviations: [MATH.md](MATH.md). GRU = gated recurrent unit, SSM = state-space model, NLMS = normalized least mean squares, KV cache = key-value cache, p95 = 95th-percentile latency, RSS = resident set size. $\sigma(x) = 1/(1+e^{-x})$; $\odot$ is the elementwise product; $x_t \in \mathbb{R}^{16}$ is the embedding row of byte $b_t$; $h_t$ is the hidden state.

### Embedding and next-byte head (`zerolm-engine/src/lib.rs`)

$$x_t = E[b_t], \qquad E \in \mathbb{R}^{256 \times 16}, \qquad E_{ij} \sim \mathcal{U}(-1, 1)$$

$$\ell_t = H h_t + c, \qquad H_{ij} \sim \mathcal{U}\!\left(-\tfrac{0.1}{\sqrt d}, \tfrac{0.1}{\sqrt d}\right), \qquad c = 0, \qquad p(b_{t+1} = v \mid b_{\le t}) = \operatorname{softmax}(\ell_t)_v$$

The head scale $0.1/\sqrt d$ keeps untrained logits near 0. Every byte then gets probability about $1/256$, so an untrained model scores about $\log_2 256 = 8$ bits per byte (measured: 8.0).

### Candidate cells (`zerolm-nn/src/`)

Every weight matrix uses fan-in initialization, $W_{ij} \sim \mathcal{U}(-1/\sqrt{n_\text{in}},\, 1/\sqrt{n_\text{in}})$. Each pre-activation $\sum_j W_{ij} x_j$ then has variance $\tfrac{1}{3}\lVert x\rVert^2 / n_\text{in}$, which stays $O(1)$ at any width when the inputs are $O(1)$.

**GRU** (selected, `gru.rs`). With update gate $z$, reset gate $r$, and candidate $\tilde h$:

$$z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z), \qquad r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$$

$$\tilde h_t = \tanh\!\big(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h\big), \qquad h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t$$

The biases are $b_z = \operatorname{linspace}(-4, 0, d)$ and $b_r = b_h = 0$. Unit $i$ keeps the fraction $1 - z_i$ of its state each step. With small inputs, $z_i \approx \sigma(b_{z,i})$, which lies between $\sigma(-4) = 0.018$ and $\sigma(0) = 0.5$. The units therefore start with memory time constants $\tau_i \approx 1/z_i$ from about 2 to about 55 bytes: a spread of timescales set by initialization, not learned. Since $h_t$ is a convex combination of $h_{t-1}$ and $\tilde h_t \in (-1, 1)$, the state stays in $(-1, 1)$ from $h_0 = 0$. Parameter count: $3de + 3d^2 + 3d$, which is 20,202 for $e = 16$, $d = 74$.

**Diagonal SSM** (`ssm.rs`). Decay and drive depend only on the input:

$$a_t = \sigma(W_a x_t + b_a), \qquad u_t = (1 - a_t) \odot \tanh(W_v x_t), \qquad h_t = a_t \odot h_{t-1} + u_t$$

with $b_a = \operatorname{linspace}(0, 4, d)$, so the initial decays $\sigma(b_a)$ run from 0.5 to 0.98. Each step is an affine map $h \mapsto A \odot h + U$, and two steps compose as

$$(A_2, U_2) \circ (A_1, U_1) = (A_2 \odot A_1,\; A_2 \odot U_1 + U_2).$$

`scan_chunked` uses this rule to fold a whole chunk into one map. The result equals token-by-token stepping up to float reassociation (tested to $10^{-5}$). Because $|u_t| \le 1 - a_t$, $|h_{t-1}| \le 1$ implies $|h_t| \le a_t + (1 - a_t) = 1$.

**Fast-weight cell** (`fastweight.rs`). A leaky state $h \in \mathbb{R}^d$ plus an associative matrix $M \in \mathbb{R}^{d \times d_k}$, $d_k = \lfloor d/2 \rfloor$, both session state:

$$q_t = \tanh(W_q x_t), \qquad \rho_t = M_{t-1}\, q_t \quad \text{(read before this step's write)}$$

$$h_t = (1 - \zeta) \odot h_{t-1} + \zeta \odot \tanh(W_x x_t + U h_{t-1} + R \rho_t + b), \qquad \zeta = \sigma(\operatorname{linspace}(-4, 0, d))$$

$$k_t = \tanh(W_k x_t), \quad v_t = \tanh(W_v x_t), \quad M_t = \operatorname{clip}_{[-8,\,8]}\!\left(0.995\, M_{t-1} + 0.5\, \frac{(v_t - M_{t-1} k_t)\, k_t^\top}{k_t^\top k_t + 10^{-6}}\right)$$

The write is the normalized delta rule (NLMS). Ignoring decay and $\epsilon$, the new readout of the written key is $M_t k_t = M_{t-1}k_t + 0.5\,(v_t - M_{t-1}k_t)$: the error on that key halves. A learning rate of 1 would store $v_t$ exactly, but it would overwrite older associations that share key directions; 0.5 trades exactness for less interference. The clamp keeps the state finite.

**Hybrid** (`hybrid.rs`, comparison arm only). A GRU plus attention over a ring of the last 8 hidden states $m_1, \dots, m_8$:

$$q = W_q h_{t-1}, \qquad \alpha_j = \operatorname{softmax}_j\!\left(\frac{q^\top m_j}{\sqrt d}\right), \qquad h_t = \tanh\!\Big(\mathrm{GRU}(h_{t-1}, x_t) + W_o \textstyle\sum_j \alpha_j m_j\Big)$$

**Budget matching.** `build_with_budget` picks the hidden width $d^\star = \arg\min_{4 \le d \le 4096} \lvert \text{params}(d) - B \rvert$ for budget $B$, which puts all four cells within 5% of 20,000.

### Bake-off metrics (`zerolm-eval/src/lib.rs`)

**Sensitivity and retention horizon.** Two runs differ only in the first byte (`0x41` and `0x14`) and then read the same $L$ random bytes:

$$S(L) = \frac{\lVert h^{a}_L - h^{b}_L \rVert_2}{\lVert h^{a}_L\rVert_2 + \lVert h^{b}_L\rVert_2 + 10^{-12}} \in [0, 1]$$

The horizon $H$ is the largest $L \in \{1, 2, 4, \dots, 256\}$ reached before $S(L)$ first falls below $10^{-3}$, taking the median over 3 seeds. It measures how long the first byte still changes the state; it says nothing about whether a learner can use that information.

**Stability.** Over 200,000 random bytes the state must stay finite, keep a constant length, and satisfy $\max_t \lVert h_t\rVert_\infty \le 16$.

**Selection score.** Over stable, eligible candidates:

$$\text{score} = 0.60\, \frac{\log_2(H + 1)}{\log_2 257} + 0.25 \left(1 - \frac{\text{state bytes}}{\max\ \text{state bytes}}\right) + 0.15\, \frac{\text{trace points}}{\max\ \text{trace points}}$$

GRU, worked through: $H = 128$; a 296-byte state against a maximum of 15,840; 3 trace points against a maximum of 4. That gives $0.60 \cdot 7.011/8.006 + 0.25 \cdot 0.981 + 0.15 \cdot 0.75 = 0.525 + 0.245 + 0.113 = 0.883$. The logarithm makes every doubling of the horizon worth the same. The p95 latency budget (2 ms per token) is a hard filter, not a score term, because wall-clock time is noisy.

## What this does and does not show
- Metrics are taken **at seeded initialization; nothing is trained or learned**. The retention horizon is the largest lag at which the hidden state still differs measurably (relative difference >= 1e-3) when only the first byte changes. It shows how long information *can* persist, not that a later learner can use it.
- GRU and fast_weight tie on horizon; GRU wins on compactness (a 296-byte state vs 15.8 KB, because M is state). The choice is sensitive to the weights in the score. With horizon alone the hybrid would lead, but it is a comparison arm by the roadmap.
- The fast-weight cell is **not evaluated as a memory** here. Phase 5 measures memory separately (`docs/MEMORY_FORMAT.md`). Selecting the GRU as the Phase 3-4 core does not rule out adding the delta-rule memory later.
- Chronological learning, held-out generalization, and forgetting cannot be measured until Phase 4.
- Bits per byte of the untrained models is about 8.0, as expected for random weights.
- The independent kernel reference is a naive `f64` implementation in Rust tests, not Python (the project has no Python).
- Linux was run locally; macOS builds are exercised by the CI matrix only.
