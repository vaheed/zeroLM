# Neural ISA

Status: uniform-engine milestone, in progress. Roadmap section 3.7; constitution invariant 10 (uniform execution).

**What "ISA" means.** ISA stands for *instruction set architecture*, a term from processor design: the fixed list of instructions a CPU can execute, which every program is ultimately built from. ZeroLM borrows the idea for a neural network. The **neural ISA** is the fixed list of tensor operations (matrix-vector products, elementwise gates, normalization, phasor binding, top-k selection, hashed row selection, the delta-rule write, and fixed repetition) that every computation of the model must be expressed in. The name is this project's own, not an external standard, which is why it does not turn up in searches. The closest published ideas are RASP and Tracr, which compile programs into Transformer weights (roadmap section 2.8).

Other abbreviations on this page: WSP = weight-space programming (compiling algorithms into weights; the `zerolm-wsp` crate); FNV-1a = Fowler-Noll-Vo hash variant 1a; SHA-256 = Secure Hash Algorithm (256-bit); IEEE-754 = the floating-point standard; NUL = the zero byte. Glossary: [math.md](math.md).

**Why it exists.** In the memory and reasoning components, facts were stored in tensors, but the decisions (parsing, search, accept or abstain) were made by ordinary Rust code. Rust code is a hand-written program, not a network. With an ISA, a capability is data: a program graph plus weights, which can be exported, ablated weight by weight, verified by a second interpreter, and later adapted by learning.

A capability of the mainline model is a **program** over a small fixed set of tensor operations plus the **weights** it runs on. Host code binds tensors, runs the program, and moves bytes in and out. It does not branch on token content or tensor values, and it does not match strings in the model's input.

- Engine interpreter: `crates/zerolm-engine/src/isa/` (`run`, `Program`, `Instr`).
- Program bundle (export format): `crates/zerolm-engine/src/isa/bundle.rs`.
- Reference interpreter: `crates/zerolm-eval/src/isa_ref.rs`. It is written separately, parses bundles from raw bytes, and shares no code with the engine. Parity is required **bit for bit**.

## Values

Every value is a dense row-major `f32` tensor `rows x cols`. A vector is `1 x n`. Indices (from `topk`) are exact integers stored as `f32`; every index in use is below 2^24.

A program declares three kinds of tensors:

| Kind | Meaning |
|---|---|
| `inputs` | Supplied per run. `rows: null` lets the caller choose the row count (for example one row per query word). |
| `weights` | Read-only: compiled circuit constants (program weights) and bound model tensors (memory keys, values, embeddings). |
| `state` | Mutable tensors written only by `delta_write`, returned after the run. |

Bound weights must be finite (checkpoint loads already reject non-finite tensors). The engine relies on this to skip zero entries of `x` in a transposed `matvec`, which is exact for finite weights; the reference interpreter does not skip.

Instructions write **registers**. An instruction may overwrite a register (needed inside `step`), but never a declared tensor. `Program::validate` checks that every operand is defined before it is read, that `delta_write` targets state, and that every output is written.

**No floating-point immediates.** Every constant (threshold, temperature, learning rate, epsilon) is a `1 x 1` (or larger) weight tensor. A program's JSON is pure structure; its numbers are in its weights.

## Operations

`acc` below starts at `+0.0` and adds terms in the stated order with IEEE-754 `f32` round-to-nearest. `exp`, `tanh`, and `sqrt` are the platform's `f32` functions; both interpreters call the same ones, so parity is exact on one platform. Cross-platform results may differ in the last bit of transcendental functions.

| Op | Family | Semantics |
|---|---|---|
| `matvec {w, x, out}` | matvec | `w: R x C`, `x: 1 x C`. `out[r] = acc over c = 0..C of w[r][c] * x[c]`; `out: 1 x R`. |
| `matvec {transpose: true}` | matvec | `x: 1 x R`. For r = 0..R in order, for each c: `out[c] += w[r][c] * x[r]`; `out: 1 x C`. |
| `matmul {a, b, out}` | matvec | `a: M x K`, `b: N x K`. `out[i][j] = acc over k of a[i][k] * b[j][k]`; `out: M x N` (that is `a b^T`). |
| `matmul {a, b, out, ab: true}` | matvec | `a: M x K`, `b: K x N`. `out[i][j] = acc over k of a[i][k] * b[k][j]`; `out: M x N` (that is `a b`). |
| `gate {f, a, b?, out}` | gate | Binary `add sub mul max min ge` (`ge`: 1.0 if a >= b else 0.0). Unary `sigmoid` (`1/(1+exp(-x))`), `tanh`, `relu`, `exp`, `neg`, `copy`. Binary operands broadcast per dimension: each dimension must be equal or 1; the output takes the larger. |
| `reduce {f, x, axis, out}` | gate | `sum` (acc), `max` (from -inf), `min` (from +inf), over `rows` (one value per row, `R x 1`), `cols` (`1 x C`), or `all` (`1 x 1`, row-major order). |
| `softmax {x, out}` | gate | Per row: `m = max`, `e_j = exp(x_j - m)`, `s = acc of e_j`, `out_j = e_j / s`. |
| `scan {x, out, exclusive}` | gate | Per row prefix product from 1.0; `exclusive` writes the product before multiplying. |
| `reshape {x, rows, cols, out}` | gate | Same data, new shape (sizes must match). |
| `transpose {x, out}` | gate | `out[c][r] = x[r][c]`; `out: C x R`. Pure data movement, so with `matmul` (which computes $AB^\top$) it also gives $AB$. |
| `norm {kind, x, eps, out}` | norm | Per row `ss = acc of x_j * x_j`; `l2`: `den = sqrt(ss + eps)`; `rms`: `den = sqrt(ss / C + eps)`; `out_j = x_j / den`, or 0 when `den == 0`. |
| `bind {a, b, out, conj}` | bind | Rows are `[re_0..re_d, im_0..im_d]`. `b` has the same rows or one row (broadcast). `out = a * b` (complex, elementwise) or `a * conj(b)`: `re = ar*br - ai*bi`, `im = ar*bi + ai*br` with `bi` negated for `conj`. |
| `topk {x, k, idx?, val?, mask?}` | topk | Per row, k times: the untaken column with the largest value; ties go to the lowest index (strict `>`). `idx`, `val`: `R x k` in selection order; `mask`: `R x C` with 1.0 at selected columns. |
| `hash_rows {bytes, table, n, out}` | hash_rows | Each row of `bytes` holds exact byte values 0..=255. For each window of `n` consecutive bytes in order, add row `fnv1a_64(window) mod table.rows` of `table`. `out: R x table.cols`. Fewer than `n` bytes gives zeros. |
| `delta_write {m, k, v, lr, decay, eps}` | delta_write | `m: R x C` state, `k: 1 x C`, `v: 1 x R`. `kk = acc of k_c^2`; `scale = lr / (eps + kk)`; `err_r = v_r - acc over c of m[r][c] * k_c` (all from the old `m`); then `m[r][c] = decay * m[r][c] + (scale * err_r) * k_c`. |
| `step {n, body}` | step | Run `body` exactly `n` times (n >= 1). The count is structure, never data. |

## Bundle format

```text
"ZLMPROG1" | u32 LE header_len | header JSON | f32 LE tensor data | SHA-256 of all previous bytes
```

The header is `{"program": <Program>, "tensors": [{"name", "rows", "cols"}, ...]}`; tensor data follows in directory order. The engine verifies the checksum on decode.

## Tests

- `crates/zerolm-engine/src/isa/` unit tests: validation, kernel semantics, delta rule, binding inverse, `hash_rows`, `step`.
- `crates/zerolm-eval/tests/isa_parity.rs`: a program using every operation, with broadcasting, a `step` loop, and a `delta_write` state, runs on both interpreters over 20 seeds; every output and the final state must match bit for bit.

## Why the zero-skip is exact

The engine's transposed `matvec` computes $o_c = \sum_r w_{rc}\, x_r$ left to right from $o_c = +0$, but it skips every $r$ with $x_r = 0$. For finite $w_{rc}$, a skipped term is $w_{rc} \cdot 0 = \pm 0$. In IEEE-754 round-to-nearest arithmetic, $a + (\pm 0) = a$ for every $a \ne 0$, and $(+0) + (-0) = +0$. The sum $u + v$ is $-0$ only when both $u$ and $v$ are $-0$, so an accumulator that starts at $+0$ can never become $-0$. Skipping zero terms therefore leaves every partial sum bit-identical. The reference interpreter does not skip, and the parity tests confirm the equality on every gate query.

Both forms of `matmul` use the same trick. The engine loops $i \to k \to j$ and skips $a_{ik} = 0$. For a fixed output element $(i, j)$ the surviving terms are still added in increasing $k$, so the order of every per-element sum is unchanged. Provenance and gather matrices are mostly zero, which is what makes the reach circuit affordable.

## Masking with a large finite penalty

Programs exclude entries by adding a penalty instead of branching. With a mask $\mu_i \in \{0, 1\}$ and $B = 10^{30}$:

$$\tilde s_i = s_i\, \mu_i - (1 - \mu_i)\, B$$

For $\mu_i = 1$ this is exactly $s_i$, since $s_i \cdot 1 = s_i$ and $s_i - 0 = s_i$. For $\mu_i = 0$ it is $-B$ (the sign of $s_i \cdot 0$ does not matter after subtracting $B$). After scaling by $\beta = 32$, $e^{\beta(-B) - \max}$ underflows to exactly 0, so excluded entries get zero softmax weight. $B$ is finite on purpose: $\beta B = 3.2 \cdot 10^{31} < 3.4 \cdot 10^{38}$, the largest `f32`, so no infinity or NaN (not a number) appears. A min-reduction over masked entries uses $+B$ the same way.

**Selecting a value without branching.** If a condition $c \in \{0, 1\}$ chooses between a value $v$ and a default $-1$: $\;c\, v + (c - 1)$. That gives $v$ when $c = 1$ and $-1$ when $c = 0$, exactly, because $v \cdot 1 = v$ and $v \cdot 0 = \pm 0$.

## The memory-read circuit

`model/prog/memory-read.json`, compiled by `zerolm_wsp::circuits::attention_read`. Inputs: the pooled query key $q \in \mathbb{R}^{128}$ and query word vectors $U \in \mathbb{R}^{n' \times 128}$. Bound model tensors: keys $K$, key words $\mathcal{K}$, values $V$, metadata $\Pi$, and byte codes $E$ (see [memory-format.md](memory-format.md)). Every step below is one ISA instruction or a few; $[\![\cdot]\!]$ is the `ge` gate.

**select** (block constants: the status-column selector $e_\text{st}$ and the window bounds 0.5, 1.5):

$$\pi = \Pi\, e_\text{st}, \qquad \mu_i = [\![\pi_i \ge 0.5]\!] - [\![\pi_i \ge 1.5]\!], \qquad h_1 = [\![\textstyle\sum_i \mu_i \ge 1]\!], \quad h_2 = [\![\textstyle\sum_i \mu_i \ge 2]\!]$$

Status values are the exact integers 0 to 3, so $\mu_i = 1$ exactly for active slots ($\pi_i = 1$).

**attend** (constants: $\beta = 32$, $w_\min = 10^{-7}$):

$$\tilde s = (K q) \odot \mu - (1 - \mu) B, \qquad (i^\star, s^\star) = \mathrm{top}_1(\tilde s), \qquad s^{(2)} = \min \mathrm{top}_2(\tilde s)$$

$$\alpha = \mathrm{softmax}(\beta \tilde s), \qquad r = V^\top \big(\alpha \odot [\![\alpha \ge w_\min]\!]\big), \qquad \text{score} = s^\star h_1 + (h_1 - 1), \qquad \text{runner-up} = s^{(2)} h_2 + (h_2 - 1)$$

The best slot's one-hot mask $e_{i^\star}$ (from `topk`) gathers its rows by transposed `matvec`: metadata $\Pi^\top e_{i^\star}$ and key words $\mathcal{K}^\top e_{i^\star}$.

**verify** (constants: thresholds 0.3 and 0.5, the key-word-count selector, and the position index $\iota = (0, 1, \dots, 7)$):

$$n_w = (\Pi^\top e_{i^\star})_\text{kw}, \qquad \kappa = \mathrm{reshape}_{8 \times 128}(\mathcal{K}^\top e_{i^\star}), \qquad g_j = \max_i (\kappa U^\top)_{ji}$$

$$\text{cov} = h_1 \cdot [\![n_w \ge 1]\!] \cdot \min_j \big(g_j + [\![\iota_j \ge n_w]\!]\, B\big)$$

Unused word positions $j \ge n_w$ get $+B$ and drop out of the minimum.

**decode** (constants: the score scale 2 and the margin selector $(1, -1)$):

$$X = \mathrm{reshape}_{48 \times 16}(r), \qquad S = 2\, X E^\top - \mathbf{1}\,\big(\lVert E_b\rVert^2\big)_b^\top, \qquad \hat b_p = \mathrm{argmax}_b S_{pb}$$

$$\nu_p = [\![\hat b_p \ge 1]\!], \qquad \text{keep}_p = \prod_{q \le p} \nu_q, \qquad \text{visited}_p = \prod_{q < p} \nu_q$$

`keep` (inclusive `scan`) marks the bytes before the first NUL; `visited` (exclusive `scan`) also includes the NUL position itself, matching where the old decoder stopped. The decoded bytes are $\hat b \odot \text{keep} \cdot h_1$, and the margin is $h_1 \cdot \min_p\big(S_{p,(1)} - S_{p,(2)} + (1 - \text{visited}_p)\, B\big)$.

**accept:**

$$a = [\![\text{score} \ge 0.3]\!] \cdot [\![\text{cov} \ge 0.5]\!] \cdot \max_p \text{keep}_p \cdot h_1, \qquad \text{value} = a \cdot \text{decoded}$$

$\max_p \text{keep}_p = \text{keep}_1$, because `keep` never increases, so this factor says the value is non-empty. The host turns `value` into bytes and stops at the first 0. A rejected read therefore comes out as an empty answer, and the host makes no decision of its own.

**Blocks and ablation.** Zeroing `verify` sets both thresholds to 0 and the word selector to 0 (so $n_w = 0$ and $\text{cov} = 0 \ge 0$). Every non-empty read is then accepted: exact answers survive and every negative is answered (measured: recall 1.000, negative false accepts 1.000). Zeroing `select` makes $\mu = 0$; zeroing `attend` sets $\beta = 0$ (a uniform average of all values); zeroing `decode` sets the score scale to 0, so every position decodes to the smallest-norm code. Each of these three drops recall to 0.

## The relational circuits (roadmap C5 and C6)

Reasoning stores each statement as a bundle of role-filler bindings, one row (a **slot**) of `memory.relations.slots` (see [reasoning.md](reasoning.md#mathematics)). Write $\sigma_s \in \mathbb{C}^d$ for slot $s$, $c_i$ for codebook symbol $i$, and $\mathrm{sim}(a, b) = \frac1d \mathrm{Re}\langle a, \bar b\rangle$. Below, $S$ is the number of used slots and $N$ the number of used symbols. The host binds only those used prefixes, so the shapes are fixed for a given memory state.

### C5: `relation-graph` (once per memory state)

$$\Sigma_{s,i} = \mathrm{sim}(\sigma_s,\ \text{subj} \circledast c_i) \qquad (S \times N,\ \text{one } \mathtt{matmul})$$

($\Sigma$ is the circuit output `subj_sim`, $O$ is `obj_t`.)

$$O_{s,j} = [\![\, j = \arg\max_k \cos(\sigma_s \oslash \text{obj},\, c_k) \,]\!] \cdot [\![\, \max_k \cos(\cdot) \ge 0.3 \,]\!]$$

$$\text{pos}_s = [\![\text{active}_s \ge 0.5]\!]\, [\![\text{polarity}_s \ge 0.5]\!], \qquad \text{neg}_s = [\![\text{active}_s \ge 0.5]\!]\, [\![-\text{polarity}_s \ge 0.5]\!]$$

This is the slow, dense step: about $2 \cdot S \cdot N \cdot 2d$ multiply-adds, roughly 85 ms for 240 slots and 240 symbols. The reasoner caches the result and recomputes it only after a write.

### C6: `relation-reach` (per question)

Given a start vector $x$, a target $y$, and a relation vector $r$ (for example the atom *isa*, or *gt* bound with *height*):

$$E_{s,i} = [\![\, \Sigma_{s,i} + \mathrm{sim}(\sigma_s, \text{rel} \circledast r) \ge 2 - 0.5 \,]\!] \cdot \text{pol}_s$$

So $E_{s,i} = 1$ when slot $s$ is an $r$-edge leaving symbol $i$. It is the reasoner's two-pair probe evaluated for every symbol at once. $x$ and $y$ are cleaned up to one-hot rows $\hat x, \hat y$ (cosine at least 0.9 with a codebook row).

The recurrence keeps a frontier $f \in \{0,1\}^N$, the reached set $R$, and a premise matrix $P \in \{0,1\}^{N \times S}$, where $P_{j,s} = 1$ means slot $s$ is on the proof of $j$. It starts from $f = R = \hat x$ and $P = 0$, and runs exactly $D = 8$ steps (`step(8)`):

$$\ell = E^\top f \in \mathbb{R}^S \qquad \text{(slots leaving the frontier)}$$

$$C_{j,s} = O_{s,j}\; \ell_s\; (1 - R_j)\; \gamma_t \qquad \text{(candidate: slot } s \text{ newly reaches } j\text{)}$$

$$Q = \mathrm{top}_1\text{-mask}(C) \odot \max_s C_{\cdot, s}, \qquad P \leftarrow P + Q E P + Q, \qquad n = \max_s Q_{\cdot,s}, \qquad R \leftarrow R + n, \qquad f \leftarrow n$$

$Q$ picks one slot per new target (the lowest slot index, because ties in `topk` go to the lower index). Then $QEP$ copies the picked slot's subject's proof, and $+Q$ adds the slot itself. Because each level runs once, a node is reached for the first time at the step equal to its distance from $x$, so $\sum_s P_{j,s}$ is the length of a **shortest** proof. This is what breadth-first search computed. The answer is

$$\text{hit} = \langle R - \hat x,\ \hat y\rangle, \qquad \text{proof} = P^\top \hat y.$$

**Depth gate and operator ablation.** $\gamma_t$ is entry $t$ of the schedule $\gamma = (1, 0, \dots, 0) + \text{enable}_{op} \cdot (0, 1, \dots, 1) \in \mathbb{R}^D$, so $\gamma_1 = 1$ always and $\gamma_{t > 1} = \text{enable}_{op}$. In the program, each step reads the first entry and shifts the schedule left with a constant shift matrix. Each inference operator has its own block holding one constant $\text{enable}_{op} = 1$. Zeroing that block limits the operator to direct, one-edge answers, which is exactly the reasoning benchmark's ablation semantics. The host passes a one-hot operator selector; that is structure, not content.

**Cost.** Each step does about six elementwise passes over $N \times S$ plus a sparse `matmul`. One reach costs about 9 ms at $N \approx S \approx 240$.

## Compiled circuits

| Circuit | Spec | Replaces | Verified by |
|---|---|---|---|
| `memory-read` | `model/prog/memory-read.json` (`zerolm_wsp::circuits::attention_read`) | memory pooled attention, coverage verification, value cleanup, accept decision | memory gate unchanged; `reference_interpreter_parity` (every gate query, bit for bit); `circuit_block_ablation` per block |
| `relation-graph` (C5) | `model/prog/relation-graph.json` (`zerolm_wsp::circuits::relational`) | per-question probe, unbind, and cleanup loops over slots | reasoning gate unchanged |
| `relation-reach` (C6) | `model/prog/relation-reach.json` | breadth-first search in class inclusion, transitivity, comparison, temporal order, and the inclusion part of negation | reasoning gate unchanged, including every operator ablation through its `enable` block |

`zerolm-wsp check` recompiles every spec and compares with `model/prog/pinned.json`.

## Not yet done

- The query encoder (word split, `#word#` trigram hashing) still runs on the host; `hash_rows` exists for it.
- Write-time bookkeeping and fast-weight recall cleanup are host code.
- Reasoning: the search (reachability and shortest proofs) runs in C6, and slot decoding in C5. Still Rust control flow: each operator's decision logic (combining yes/no proofs, the inheritance level scan, the negation edge scan, conditional rule chaining, count arithmetic), the one-step `edges` probes they use, and the bounded-language parser.
- No circuit has been through the byte-permutation test yet.
