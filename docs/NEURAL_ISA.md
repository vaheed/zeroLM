# Neural ISA

Status: Phase 7, in progress. Roadmap section 3.7; constitution invariant 11 (uniform execution).

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

## Not yet done (Phase 7 remainder)

The ISA exists and is verified, but Phase 5 retrieval and the Phase 6 parser and operators still run as Rust control flow until they are recompiled (see `docs/CLAIMS_AND_LIMITATIONS.md`).
