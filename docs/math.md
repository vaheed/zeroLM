# Mathematical reference and glossary

This page fixes the notation used across the ZeroLM docs, explains every abbreviation, and points to where each formula is derived. Formulas are written in LaTeX (GitHub renders `$...$` and `$$...$$`). Each formula states exactly what the code computes; the file that implements it is named next to it in the topic doc.

## Glossary

| Term | Stands for | Meaning in ZeroLM |
|---|---|---|
| **ISA** | Instruction set architecture | Borrowed from CPU design, where an ISA is the fixed list of instructions a processor can run. The **neural ISA** is ZeroLM's fixed list of tensor operations (matvec, gate, norm, bind, top-k, hash_rows, delta_write, step) that every model computation must be built from. The name is coined by this project; see [neural-isa.md](neural-isa.md). |
| **WSP** | Weight-space programming | Writing an algorithm directly into network weights instead of learning it. In ZeroLM, WSP is a *compiler* (`zerolm-wsp`) from small circuit specifications to weights; it never writes language data. |
| **Weight groups** | | Named sets of tensors, each with its own initialization source and update rule (`model/specs/weight-groups.json`): core weights (recurrent core), embedding weights, fast weights (short-lived associations), memory weights (persistent facts), skill weights, meta weights (update-rule parameters), and program weights (compiled circuit constants, tensors `prog.<circuit>.<name>`, group id `w_prog`). |
| **Circuit** | | A compiled program plus its constants (program weights) that implements one algorithm, for example `memory-read`. Its **blocks** are named groups of constants that can be ablated. |
| **Ablation** | | Disabling or zeroing a component to test causally whether behavior depends on it. |
| **GRU** | Gated recurrent unit | The selected recurrent cell ([architecture.md](architecture.md)). |
| **SSM** | State-space model | Here, a diagonal gated linear recurrence (a bake-off candidate). |
| **KV cache** | Key-value cache | The growing store of past keys and values a Transformer keeps. ZeroLM has none: its state has a fixed size. |
| **BPB** | Bits per byte | Average code length of the next byte under the model, $-\frac{1}{N}\sum_t \log_2 p(b_t \mid b_{<t})$. Lower is better; 8 means chance over 256 bytes. |
| **ECE** | Expected calibration error | How far the model's top-1 confidence is from its actual accuracy ([online-learning.md](online-learning.md)). |
| **NLMS** | Normalized least mean squares | The adaptive-filter update $\Delta W = \eta\, e\, x^\top / (\lVert x\rVert^2 + \epsilon)$. ZeroLM's "normalized delta rule" is NLMS. |
| **Delta rule** | | A local learning rule: change a weight by (target minus prediction) times the input. |
| **Hebbian rule** | | A local rule with no error term: strengthen connections between co-active units ("fire together, wire together"). |
| **BPTT** | Backpropagation through time | Gradient training of a recurrent network unrolled over time. Not used in the mainline. |
| **FHRR** | Fourier holographic reduced representations | A vector-symbolic algebra with complex unit phasors, where binding is elementwise multiplication ([reasoning.md](reasoning.md)). |
| **VSA** | Vector-symbolic architecture | The family of algebras (FHRR is one) that encode structures as high-dimensional vectors. |
| **Phasor** | | A complex number of modulus 1, $e^{i\theta}$. |
| **kWTA** | k-winners-take-all | Keep the $k$ largest entries of a vector and zero the rest; used as a sparse code in fast weights. |
| **FNV-1a** | Fowler-Noll-Vo hash, variant 1a | The 64-bit byte hash used for tags, trigram buckets, and `hash_rows`. |
| **SplitMix64** | | The integer pseudo-random generator behind every seeded initialization ([bootstrap.md](bootstrap.md)). |
| **SHA-256** | Secure Hash Algorithm, 256-bit | Used to pin weights, checkpoints, compiled circuits, and reports. |
| **PPM** | Prediction by partial matching | A classic adaptive context-mixing compressor; a planned baseline for the real-text milestone. |
| **KN-5** | Kneser-Ney smoothing, order 5 | A smoothed 5-gram language model; a planned baseline for the real-text milestone. |
| **Witten-Bell** | | The n-gram smoothing used by the online-learning byte baselines ([online-learning.md](online-learning.md)). |
| **BLiMP** | Benchmark of Linguistic Minimal Pairs | A grammaticality benchmark planned from the real-text milestone on. |
| **LAMBADA** | | A last-word prediction benchmark planned from the real-text milestone on. |
| **p50 / p95** | 50th / 95th percentile | Median and tail latency. |
| **RSS** | Resident set size | Physical memory used by the process. |
| **MSRV** | Minimum supported Rust version | 1.75 for this workspace. |
| **CI** | Continuous integration | The GitHub Actions jobs in `.github/workflows/ci.yml`. |
| **L0-L4** | Language-model levels | Capability levels defined in roadmap section 7.1; the system is at L0. |
| **Gate A / B / C** | | Pre-registered pass/fail tests for the scalable-core, local-representation, and scaling milestones (roadmap section 5). |

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
| Seeded initialization | SplitMix64, tag streams, uniform and fan-in init, FNV-1a, phasor angles | [bootstrap.md](bootstrap.md#mathematics) |
| Recurrent cells and bake-off | GRU, SSM scan composition, fast-weight cell, hybrid attention, sensitivity, horizon, selection score | [architecture.md](architecture.md#mathematics) |
| Online learning | BPB, softmax readout, delta rule as a normalized gradient step, update-norm bound, Hebbian prototypes, Witten-Bell, ECE, gate statistic | [online-learning.md](online-learning.md#mathematics) |
| Memory | trigram key encoder, attention read, coverage, nearest-code cleanup, kWTA, fast-weight delta rule, 24-bit integer packing | [memory-format.md](memory-format.md#mathematics) |
| Reasoning | FHRR bind/unbind/bundle, similarity statistics, probe threshold, cleanup, fractional-power numbers, discourse state, operator semantics | [reasoning.md](reasoning.md#mathematics) |
| Neural ISA and circuits | operation semantics, masking with a large penalty, the memory-read circuit as equations, exactness of the zero-skip | [neural-isa.md](neural-isa.md) |
