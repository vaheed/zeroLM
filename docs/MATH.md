# Mathematical reference and glossary

This page fixes the notation used across the ZeroLM docs, explains every abbreviation, and points to where each formula is derived. Formulas are written in LaTeX (GitHub renders `$...$` and `$$...$$`). Each formula states exactly what the code computes; the file that implements it is named next to it in the topic doc.

## Glossary

| Term | Stands for | Meaning in ZeroLM |
|---|---|---|
| **ISA** | Instruction set architecture | Borrowed from CPU design, where an ISA is the fixed list of instructions a processor can run. The **neural ISA** is ZeroLM's fixed list of tensor operations (matvec, gate, norm, bind, top-k, hash_rows, delta_write, step) that every model computation must be built from. The name is coined by this project; see [NEURAL_ISA.md](NEURAL_ISA.md). |
| **WSP** | Weight-space programming | Writing an algorithm directly into network weights instead of learning it. In ZeroLM, WSP is a *compiler* (`zerolm-wsp`) from small circuit specifications to weights; it never writes language data. |
| **W_PROG** | Program weights | The weight group holding compiled circuit constants (tensors `prog.<circuit>.<name>`). |
| **W_CORE, W_EMBED, W_FAST, W_MEMORY, W_SKILL, W_META** | Weight groups | Recurrent core, byte embedding, fast associative weights, persistent memory, skill modules, and update-rule metadata (roadmap section 3.3). |
| **Circuit** | | A compiled program plus its `W_PROG` constants that implements one algorithm, for example `memory-read`. Its **blocks** are named groups of constants that can be ablated. |
| **Ablation** | | Disabling or zeroing a component to test causally whether behavior depends on it. |
| **GRU** | Gated recurrent unit | The selected recurrent cell ([ARCHITECTURE.md](ARCHITECTURE.md)). |
| **SSM** | State-space model | Here, a diagonal gated linear recurrence (a bake-off candidate). |
| **KV cache** | Key-value cache | The growing store of past keys and values a Transformer keeps. ZeroLM has none: its state has a fixed size. |
| **BPB** | Bits per byte | Average code length of the next byte under the model, $-\frac{1}{N}\sum_t \log_2 p(b_t \mid b_{<t})$. Lower is better; 8 means chance over 256 bytes. |
| **ECE** | Expected calibration error | How far the model's top-1 confidence is from its actual accuracy ([ONLINE_LEARNING.md](ONLINE_LEARNING.md)). |
| **NLMS** | Normalized least mean squares | The adaptive-filter update $\Delta W = \eta\, e\, x^\top / (\lVert x\rVert^2 + \epsilon)$. ZeroLM's "normalized delta rule" is NLMS. |
| **Delta rule** | | A local learning rule: change a weight by (target minus prediction) times the input. |
| **Hebbian rule** | | A local rule with no error term: strengthen connections between co-active units ("fire together, wire together"). |
| **BPTT** | Backpropagation through time | Gradient training of a recurrent network unrolled over time. Not used in the mainline. |
| **FHRR** | Fourier holographic reduced representations | A vector-symbolic algebra with complex unit phasors, where binding is elementwise multiplication ([REASONING.md](REASONING.md)). |
| **VSA** | Vector-symbolic architecture | The family of algebras (FHRR is one) that encode structures as high-dimensional vectors. |
| **Phasor** | | A complex number of modulus 1, $e^{i\theta}$. |
| **kWTA** | k-winners-take-all | Keep the $k$ largest entries of a vector and zero the rest; used as a sparse code in W_FAST. |
| **FNV-1a** | Fowler-Noll-Vo hash, variant 1a | The 64-bit byte hash used for tags, trigram buckets, and `hash_rows`. |
| **SplitMix64** | | The integer pseudo-random generator behind every seeded initialization ([BOOTSTRAP.md](BOOTSTRAP.md)). |
| **SHA-256** | Secure Hash Algorithm, 256-bit | Used to pin weights, checkpoints, compiled circuits, and reports. |
| **PPM** | Prediction by partial matching | A classic adaptive context-mixing compressor; a planned Phase 8 baseline. |
| **KN-5** | Kneser-Ney smoothing, order 5 | A smoothed 5-gram language model; a planned Phase 8 baseline. |
| **Witten-Bell** | | The n-gram smoothing used by the Phase 4 byte baselines ([ONLINE_LEARNING.md](ONLINE_LEARNING.md)). |
| **BLiMP** | Benchmark of Linguistic Minimal Pairs | A grammaticality benchmark planned for Phase 8+. |
| **LAMBADA** | | A last-word prediction benchmark planned for Phase 8+. |
| **p50 / p95** | 50th / 95th percentile | Median and tail latency. |
| **RSS** | Resident set size | Physical memory used by the process. |
| **MSRV** | Minimum supported Rust version | 1.75 for this workspace. |
| **CI** | Continuous integration | The GitHub Actions jobs in `.github/workflows/ci.yml`. |
| **L0-L4** | Language-model levels | Capability levels defined in roadmap section 7.1; the system is at L0. |
| **Gate A / B / C** | | Pre-registered pass/fail tests for Phases 9, 10, 11 (roadmap section 5). |

## Notation

| Symbol | Meaning |
|---|---|
| $b_t \in \{0, \dots, 255\}$ | byte at position $t$ |
| $x_t$, $h_t$ | input embedding and hidden state at step $t$ |
| $W, U, M$ | matrices (row-major in code); $W_i$ is row $i$ |
| $\odot$ | elementwise (Hadamard) product |
| $\sigma(x) = 1/(1+e^{-x})$ | logistic sigmoid |
| $\mathrm{softmax}(z)_i = e^{z_i - m} / \sum_j e^{z_j - m}$, $m = \max_j z_j$ | softmax, computed with the max subtracted for stability |
| $\langle a, b\rangle = a^\top b$; $\cos(a,b) = \langle a,b\rangle / (\lVert a\rVert\lVert b\rVert)$ | dot product and cosine |
| $e_y$ | one-hot vector for index $y$ |
| $\mathcal{U}(a, b)$ | uniform distribution |
| $[\![P]\!]$ | 1 if $P$ holds, else 0 (the ISA `ge` gate computes $[\![a \ge b]\!]$) |

## Where each formula lives

| Topic | Formulas | Doc |
|---|---|---|
| Seeded initialization | SplitMix64, tag streams, uniform and fan-in init, FNV-1a, phasor angles | [BOOTSTRAP.md](BOOTSTRAP.md#mathematics) |
| Recurrent cells and bake-off | GRU, SSM scan composition, fast-weight cell, hybrid attention, sensitivity, horizon, selection score | [ARCHITECTURE.md](ARCHITECTURE.md#mathematics) |
| Online learning | BPB, softmax readout, delta rule as a normalized gradient step, update-norm bound, Hebbian prototypes, Witten-Bell, ECE, gate statistic | [ONLINE_LEARNING.md](ONLINE_LEARNING.md#mathematics) |
| Memory | trigram key encoder, attention read, coverage, nearest-code cleanup, kWTA, W_FAST delta rule, 24-bit integer packing | [MEMORY_FORMAT.md](MEMORY_FORMAT.md#mathematics) |
| Reasoning | FHRR bind/unbind/bundle, similarity statistics, probe threshold, cleanup, fractional-power numbers, discourse state, operator semantics | [REASONING.md](REASONING.md#mathematics) |
| Neural ISA and circuits | operation semantics, masking with a large penalty, the memory-read circuit as equations, exactness of the zero-skip | [NEURAL_ISA.md](NEURAL_ISA.md) |
