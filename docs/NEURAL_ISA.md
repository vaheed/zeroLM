# Neural ISA

Status: Phase 7, in progress. Roadmap section 3.7; constitution invariant 10 (uniform execution).

**What "ISA" means.** ISA stands for *instruction set architecture*, a term from processor design: the fixed list of instructions a CPU can execute, which every program is ultimately built from. ZeroLM borrows the idea for a neural network. The **neural ISA** is the fixed list of tensor operations (matrix-vector products, elementwise gates, normalization, phasor binding, top-k selection, hashed row selection, the delta-rule write, and fixed repetition) that every computation of the model must be expressed in. The name is this project's own, not an external standard, which is why it does not turn up in searches. The closest published ideas are RASP and Tracr, which compile programs into Transformer weights (roadmap section 2.8).

Other abbreviations on this page: WSP = weight-space programming (compiling algorithms into weights; the `zerolm-wsp` crate); W_PROG = the weight group of compiled circuit constants; FNV-1a = Fowler-Noll-Vo hash variant 1a; SHA-256 = Secure Hash Algorithm (256-bit); IEEE-754 = the floating-point standard; NUL = the zero byte. Glossary: [MATH.md](MATH.md).

**Why it exists.** In Phases 5-6, facts were stored in tensors, but the decisions (parsing, search, accept or abstain) were made by ordinary Rust code. Rust code is a hand-written program, not a network. With an ISA, a capability is data: a program graph plus weights, which can be exported, ablated weight by weight, verified by a second interpreter, and later adapted by learning.

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
| `weights` | Read-only: compiled `W_PROG` constants and bound model tensors (memory keys, values, embeddings). |
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
| `gate {f, a, b?, out}` | gate | Binary `add sub mul max min ge` (`ge`: 1.0 if a >= b else 0.0). Unary `sigmoid` (`1/(1+exp(-x))`), `tanh`, `relu`, `exp`, `neg`, `copy`. Binary operands broadcast per dimension: each dimension must be equal or 1; the output takes the larger. |
| `reduce {f, x, axis, out}` | gate | `sum` (acc), `max` (from -inf), `min` (from +inf), over `rows` (one value per row, `R x 1`), `cols` (`1 x C`), or `all` (`1 x 1`, row-major order). |
| `softmax {x, out}` | gate | Per row: `m = max`, `e_j = exp(x_j - m)`, `s = acc of e_j`, `out_j = e_j / s`. |
| `scan {x, out, exclusive}` | gate | Per row prefix product from 1.0; `exclusive` writes the product before multiplying. |
| `reshape {x, rows, cols, out}` | gate | Same data, new shape (sizes must match). |
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

## Masking with a large finite penalty

Programs exclude entries by adding a penalty instead of branching. With a mask $\mu_i \in \{0, 1\}$ and $B = 10^{30}$:

$$\tilde s_i = s_i\, \mu_i - (1 - \mu_i)\, B$$

For $\mu_i = 1$ this is exactly $s_i$, since $s_i \cdot 1 = s_i$ and $s_i - 0 = s_i$. For $\mu_i = 0$ it is $-B$ (the sign of $s_i \cdot 0$ does not matter after subtracting $B$). After scaling by $\beta = 32$, $e^{\beta(-B) - \max}$ underflows to exactly 0, so excluded entries get zero softmax weight. $B$ is finite on purpose: $\beta B = 3.2 \cdot 10^{31} < 3.4 \cdot 10^{38}$, the largest `f32`, so no infinity or NaN (not a number) appears. A min-reduction over masked entries uses $+B$ the same way.

**Selecting a value without branching.** If a condition $c \in \{0, 1\}$ chooses between a value $v$ and a default $-1$: $\;c\, v + (c - 1)$. That gives $v$ when $c = 1$ and $-1$ when $c = 0$, exactly, because $v \cdot 1 = v$ and $v \cdot 0 = \pm 0$.

## The memory-read circuit

`model/prog/memory-read.json`, compiled by `zerolm_wsp::circuits::attention_read`. Inputs: the pooled query key $q \in \mathbb{R}^{128}$ and query word vectors $U \in \mathbb{R}^{n' \times 128}$. Bound model tensors: keys $K$, key words $\mathcal{K}$, values $V$, metadata $\Pi$, and byte codes $E$ (see [MEMORY_FORMAT.md](MEMORY_FORMAT.md)). Every step below is one ISA instruction or a few; $[\![\cdot]\!]$ is the `ge` gate.

**select** (block constants: the status-column selector $e_\text{st}$ and the window bounds 0.5, 1.5):

$$\pi = \Pi\, e_\text{st}, \qquad \mu_i = [\![\pi_i \ge 0.5]\!] - [\![\pi_i \ge 1.5]\!], \qquad h_1 = [\![\textstyle\sum_i \mu_i \ge 1]\!], \quad h_2 = [\![\textstyle\sum_i \mu_i \ge 2]\!]$$

Status values are the exact integers 0 to 3, so $\mu_i = 1$ exactly for active slots ($\pi_i = 1$).

**attend** (constants: $\beta = 32$, $w_\min = 10^{-7}$):

$$\tilde s = (K q) \odot \mu - (1 - \mu) B, \qquad (i^*, s^*) = \operatorname{top}_1(\tilde s), \qquad s^{(2)} = \min \operatorname{top}_2(\tilde s)$$

$$\alpha = \operatorname{softmax}(\beta \tilde s), \qquad r = V^\top \big(\alpha \odot [\![\alpha \ge w_\min]\!]\big), \qquad \text{score} = s^* h_1 + (h_1 - 1), \qquad \text{runner\_up} = s^{(2)} h_2 + (h_2 - 1)$$

The best slot's one-hot mask $e_{i^*}$ (from `topk`) gathers its rows by transposed `matvec`: metadata $\Pi^\top e_{i^*}$ and key words $\mathcal{K}^\top e_{i^*}$.

**verify** (constants: thresholds 0.3 and 0.5, the key-word-count selector, and the position index $\iota = (0, 1, \dots, 7)$):

$$n_w = (\Pi^\top e_{i^*})_\text{kw}, \qquad \kappa = \operatorname{reshape}_{8 \times 128}(\mathcal{K}^\top e_{i^*}), \qquad g_j = \max_i (\kappa U^\top)_{ji}$$

$$\text{cov} = h_1 \cdot [\![n_w \ge 1]\!] \cdot \min_j \big(g_j + [\![\iota_j \ge n_w]\!]\, B\big)$$

Unused word positions $j \ge n_w$ get $+B$ and drop out of the minimum.

**decode** (constants: the score scale 2 and the margin selector $(1, -1)$):

$$X = \operatorname{reshape}_{48 \times 16}(r), \qquad S = 2\, X E^\top - \mathbf{1}\,\big(\lVert E_b\rVert^2\big)_b^\top, \qquad \hat b_p = \operatorname{argmax}_b S_{pb}$$

$$\nu_p = [\![\hat b_p \ge 1]\!], \qquad \text{keep}_p = \prod_{q \le p} \nu_q, \qquad \text{visited}_p = \prod_{q < p} \nu_q$$

`keep` (inclusive `scan`) marks the bytes before the first NUL; `visited` (exclusive `scan`) also includes the NUL position itself, matching where the old decoder stopped. The decoded bytes are $\hat b \odot \text{keep} \cdot h_1$, and the margin is $h_1 \cdot \min_p\big(S_{p,(1)} - S_{p,(2)} + (1 - \text{visited}_p)\, B\big)$.

**accept:**

$$a = [\![\text{score} \ge 0.3]\!] \cdot [\![\text{cov} \ge 0.5]\!] \cdot \max_p \text{keep}_p \cdot h_1, \qquad \text{value} = a \cdot \text{decoded}$$

$\max_p \text{keep}_p = \text{keep}_1$, because `keep` never increases, so this factor says the value is non-empty. The host turns `value` into bytes and stops at the first 0. A rejected read therefore comes out as an empty answer, and the host makes no decision of its own.

**Blocks and ablation.** Zeroing `verify` sets both thresholds to 0 and the word selector to 0 (so $n_w = 0$ and $\text{cov} = 0 \ge 0$). Every non-empty read is then accepted: exact answers survive and every negative is answered (measured: recall 1.000, negative false accepts 1.000). Zeroing `select` makes $\mu = 0$; zeroing `attend` sets $\beta = 0$ (a uniform average of all values); zeroing `decode` sets the score scale to 0, so every position decodes to the smallest-norm code. Each of these three drops recall to 0.

## Compiled circuits

| Circuit | Spec | Replaces | Verified by |
|---|---|---|---|
| `memory-read` | `model/prog/memory-read.json` (`zerolm_wsp::circuits::attention_read`) | Phase 5 pooled attention, coverage verification, value cleanup, accept decision | Phase 5 gate unchanged; `reference_interpreter_parity` (every gate query, bit for bit); `circuit_block_ablation` per block |

`zerolm-wsp check` recompiles every spec and compares with `model/prog/pinned.json`.

## Not yet done (Phase 7 remainder)

- The query encoder (word split, `#word#` trigram hashing) still runs on the host; `hash_rows` exists for it.
- Write-time bookkeeping and `W_FAST` recall cleanup are host code.
- The Phase 6 parser and the eight inference operators still run as Rust control flow; they are the next circuits (C5, C6).
- No circuit has been through the byte-permutation test yet.
