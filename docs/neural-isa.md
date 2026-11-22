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

## The uniform-engine gate

`cargo run --release -p zerolm-eval --bin zerolm-uniform` (about 4 minutes; report schema `research/schemas/uniform-engine-report.schema.json`):

| Check | What it shows |
|---|---|
| circuits pinned and reproducible | every spec in `model/prog/` compiles twice to the same SHA-256, equal to `model/prog/pinned.json` |
| patch round trip | for every circuit: a one-value patch applies, a second apply is refused, and revert restores the original hash |
| memory reference parity | 400 memory queries (taught keys and unknown keys) through `memory-encode` and `memory-read`, plus a short session of new, repeated, contradicting, corrected, confirmed, and forgotten facts, history, compaction, and fast associations through the write circuits, `fast-write` (including its final `memory.fast` state), and `fast-recall`: every run reproduced bit for bit from the exported bundles |
| reasoning reference parity | every circuit run while a generated world is taught and its questions answered, intact and with each operator ablated, replayed on the reference interpreter (3,520 runs, all bit-exact; every decision circuit included) |
| host audit | the model-path files listed in `model/prog/host-audit.json` contain no text processing; host code that still makes content decisions is listed as **pending** |

The gate passes when every check holds. The milestone counts as complete only when the pending list is also empty, which means all model behavior is in weights.

## The atom-encode circuit

`model/prog/atom-encode.json`, compiled by `zerolm_wsp::circuits::atom_encode`. It computes the vector of a symbol, role, or relation from the bytes of its namespaced name (for example `sym:cat`), replacing the former host random generator. Input: a row of $L = 128$ byte values with the name at positions $1..\ell$. Bound model tensor: the seeded random table $T$ (`core.reasoner.atom_table`, 1024 rows of $2d$ values, $d = 512$).

1. Make three copies of the row and put salt $j \in \{1, 2, 3\}$ at position 0 (constants: a column of ones and a salt matrix).
2. `hash_rows` with window $n = L$ (exactly one window per copy) gives rows $h_j = T[\mathrm{FNV1a}(\text{copy}_j) \bmod 1024]$.
3. Per complex component $k$, normalize the pair $(\mathrm{re}_k, \mathrm{im}_k)$ to unit length: $u_{j,k} = h_{j,k} / \lVert h_{j,k} \rVert$ (reshape, transpose, `norm`).
4. Bind the three: $a = u_1 \otimes u_2 \otimes u_3$ (elementwise complex product).

Two different names collide only if all three hashes collide (probability about $1024^{-3}$), and the phases of unrelated names are independent, so their cosine is about 0 with standard deviation $1/\sqrt{2d}$. The host copies bytes and memoizes the circuit's output by the same key; it inspects no byte.

## The value codec and W_FAST circuits

`model/prog/value-encode.json`, `fast-recall.json`, `fast-write.json`, compiled by `zerolm_wsp::circuits::value_codec`; they replace the host value encoder, the host nearest-code decoder, and the host fast-weight code (`retrieval.rs` and `fast_weights.rs` are deleted). Bound tensors: the byte code table $E$ (`core.embed`, 256 x 16), the separator $P$ (`core.fast_separator`), and `memory.fast` (a weight for recall, the `delta_write` state for writes).

- `value-encode`: value bytes (NUL-padded to 48) compared with the indices 0..255 (`ge` both ways) give a one-hot $O$; the row is $OE$, reshaped to 1 x 768.
- `fast-recall`: $\sigma(k) = \mathrm{topk}_{32}(Pk) / \sqrt{32}$, the read $M\sigma$, and the same decoding.
- `fast-write`: the same $\sigma$, the value encoded as in `value-encode`, then `delta_write` with learning rate 0 and decay 0.99 (decay only), and `delta_write` with learning rate 1, epsilon $10^{-6}$ and decay 1 (the delta rule on the decayed matrix).

`value-encode` remains a standalone pinned program (the write circuits encode values in-circuit with the same block). The nearest-code decoding and the byte comparison that tell an unchanged fact from a contradiction are part of `memory-write` (below).

## The memory write circuits

`model/prog/memory-{write,forget,compact,history}.json`, compiled by `zerolm_wsp::circuits::memory_write`. They decide every change to the semantic and episodic memory; the host stamps a write with what it owns (sequence number, source, provenance, key word count), refuses a write past capacity, and moves the rows the outputs specify.

- `memory-write` (teach, correct, confirm; a one-hot `op`): the same-key slot is the best active key with $k \cdot K_s \ge 0.95$; its value row is decoded (nearest code) and compared byte by byte with the new value; the free slot is the lowest empty one (`topk`, ties to the lowest index). The outcome is one-hot: teach gives unchanged, conflict, or stored; correct gives unchanged, superseded, or stored; confirm gives confirmed or unknown. Outputs: the new slot's key, key-word, value (encoded in-circuit), and metadata rows (zero unless a slot is written; a superseding record links to the old slot), the old slot's metadata after the operation (contradictions + 1, or status superseded with a forward link, or confidence confirmed, and the updated time), and the episode row (op code and slot).
- `memory-forget`: the same-key slot, then one supersession link per step for as many steps as there are slots (any chain is followed; visited slots are not revisited), the valid episodic records that refer to the chain (their keys are erased), and the metadata with the chain marked deleted.
- `memory-compact`: active slots first in index order (`topk` with $k = S$), a permutation matrix $P$ (row $j$ holds old slot $\mathrm{order}_j$), the map old $\to$ new ($-1$ for reclaimed slots), and every supersession link and episodic slot reference remapped through it; rows past the active ones become empty records.
- `memory-history`: valid episodes whose key has $k \cdot e \ge 0.95$ with the query key.

## The reasoner write circuits

`model/prog/relation-probe.json` and `model/prog/symbol-cleanup.json`, compiled by `zerolm_wsp::circuits::probe`. They decide every reasoner write; the host only appends their output rows at its slot and codebook counters (and refuses a write past capacity).

- `relation-probe` (a statement): inputs `fillers` ($K \times 2d$), one-hot `role_sel` ($K \times 5$), the metadata `record` the statement would get, `gate` (1 when the subject resolved), and the discourse state. Roles are selected from `core.reasoner.roles`, bound to the fillers, and summed into $q = \sum_k \rho_{r_k} \circledast f_k$. An active slot with the record's polarity, $K$ stored pairs, and $\mathrm{sim}(\text{slot}, q) \ge K - 0.5$ is a duplicate. `stored` $=$ gate $\cdot$ (1 $-$ duplicate); the outputs `row` $= \text{stored} \cdot q$ and `meta_row` $= \text{stored} \cdot \text{record}$ are what the host appends; `context_out` is $0.5\,c + s$ (s the subject filler) when gate is 1, else $c$. With no slots yet, the host binds one all-zero inactive row, so no tensor is empty.
- `symbol-cleanup` (a name or "it"): input $v$, a one-hot `mode`, and `gate`. Cosines $c_i = C_i \cdot \hat v / \sqrt d$ with every codebook row, the best row, and `found` when $c \ge$ the mode's threshold: 0.9 (the same symbol) or 0.4 (resolving "it" from the discourse state; a zero state never resolves). `added` $=$ gate $\cdot$ (1 $-$ found) and `row` $=$ added $\cdot v$: interning appends that row. An empty codebook is bound as one zero row.

The binding ablation binds identity role phasors; the discourse-state ablation keeps the state at zero.

## The decision circuits

Compiled by `zerolm_wsp::circuits::decide` (`model/prog/relation-{join,levels,rules,counts,verdict}.json`). They replace the operators' Rust control flow: each is a fixed number of steps over the C5 outputs (`subj_sim`, `obj_t`, `pos`, `neg`), with $E_r[s, i] = [\mathrm{subj\_sim}[s,i] + \mathrm{sim}(\text{slot}_s, r \circledast \rho_{\text{rel}}) \ge 1.5] \cdot \mathrm{pol}[s]$ the r-edges leaving symbol i. Proofs are slot masks or premise matrices (row j = slots on the proof of j); among candidates the shortest proof wins (score $10^6 - 1 - \text{length}$, ties to the lowest slot).

| circuit | operator | computation |
|---|---|---|
| `relation-join` | negation | the shortest negative r-edge from $\{x\} \cup$ anc(x) to $\{y\} \cup$ anc(y) (ancestors and their premise matrices from two C6 runs; with class inclusion disabled, only x and y) |
| `relation-levels` | attribute inheritance | breadth-first levels over positive isa edges; at the first level whose frontier has a positive or negative can-edge to v, both are latched ($\mathrm{done} \leftarrow \max(\mathrm{done}, \mathrm{found})$) |
| `relation-rules` | conditionals | backward levels from property p over in-scope rules (scope "something" or x, cosine $\ge 0.3$), latched at the first level where x has a frontier property directly |
| `relation-counts` | arithmetic | breadth-first over "more" edges carrying $\mathrm{num}(\sum \delta)$ by binding cleaned differences; each reached node's count is cleaned up, bound to its carried sum, and decoded; one distinct value is the answer, several are a contradiction |
| `relation-verdict` | combination | $(p, n) \mapsto$ yes $p\bar n$, no $\bar p n$, contradiction $pn$, unknown $\bar p \bar n$, and the union of the supporting proofs |

Each operator's enable constant is its own block, so zeroing it is the operator ablation. The host chooses which circuits a question's parsed form runs and labels the result from proof lengths and levels; it makes no content decision.

## The number circuits

`model/prog/number-encode.json` and `model/prog/number-decode.json`, compiled by `zerolm_wsp::circuits::number`. They replace the host's fractional-power number encoder and its cleanup scan. Numbers are $\mathrm{num}(n)_k = e^{i n \phi_k}$ with a seeded base $\phi$, and two seeded tables are initialized from it (bound model tensors, verified on load): the digit table $D$ (`core.reasoner.number_digits`, 70 rows, row $10q + v = \mathrm{num}(v\,10^q)$) and the codebook $C$ (`core.reasoner.number_codebook`, 2,001 rows, row $j = \mathrm{num}(j - 1000)$).

**`number-encode`.** Input: a row of $L = 8$ byte values holding the decimal numeral (optional leading `-`) at positions $0..\ell$. Because $\mathrm{num}$ is a homomorphism from addition to binding, $\mathrm{num}(\sum_q v_q 10^q) = \circledast_q\, \mathrm{num}(v_q 10^q)$:

1. `classify`: compare each byte with the codes 48..57 (`ge` both ways, `mul`): a digit one-hot $O \in \{0,1\}^{L \times 10}$ and the digit indicator $s = O\mathbf{1}$.
2. `position`: the place of byte $j$ is the number of digits after it, $p = s^\top U$ with $U_{ij} = [i > j]$; compared with the places $0..6$ it gives a one-hot $P \in \{0,1\}^{L \times 7}$.
3. `select`: $S = (P\,\mathrm{Spread}) \odot (O\,\mathrm{Tile})$ has a single 1 at column $10q + v$ in each digit row, so $SD$ holds $\mathrm{num}(v_j 10^{q_j})$ per digit byte.
4. `fill`: non-digit bytes (padding, the sign) get the identity phasor $\mathrm{num}(0)$: add $(1 - s)\,e$ with $e = [1, \dots, 1, 0, \dots, 0]$.
5. `product`: bind the $L$ rows.
6. `sign`: if byte 0 equals 45, negate the imaginary half (conjugation, $\mathrm{num}(-n) = \overline{\mathrm{num}(n)}$).

A digit beyond the seventh place selects no row, which zeroes the output; the parser only produces integers up to $10^6$. The output equals the host formula up to `f32` rounding of the products (similarity $1 \pm 10^{-4}$, unit test).

**`number-decode`.** Input: a vector $v$ (1 x $2d$). `matvec` scores $Cv$, `topk` keeps the best row (ties to the lower index), and the mask reads out the integer (a constant column $-1000..1000$, exact in `f32`) and the clean row of $C$. The arithmetic operator binds clean rows and decodes again, so a sum is exactly `decode(num(a) * num(b))`.

Ablation (unit test): zeroing `sign` zeroes every output; zeroing `position` makes each digit select all of its places (cosine with the true vector about 1/7); zeroing `readout` reads every number as 0.

## Why the zero-skip is exact

The engine's transposed `matvec` computes $o_c = \sum_r w_{rc}\, x_r$ left to right from $o_c = +0$, but it skips every $r$ with $x_r = 0$. For finite $w_{rc}$, a skipped term is $w_{rc} \cdot 0 = \pm 0$. In IEEE-754 round-to-nearest arithmetic, $a + (\pm 0) = a$ for every $a \ne 0$, and $(+0) + (-0) = +0$. The sum $u + v$ is $-0$ only when both $u$ and $v$ are $-0$, so an accumulator that starts at $+0$ can never become $-0$. Skipping zero terms therefore leaves every partial sum bit-identical. The reference interpreter does not skip, and the parity tests confirm the equality on every gate query.

Both forms of `matmul` use the same trick. The engine loops $i \to k \to j$ and skips $a_{ik} = 0$. For a fixed output element $(i, j)$ the surviving terms are still added in increasing $k$, so the order of every per-element sum is unchanged. Provenance and gather matrices are mostly zero, which is what makes the reach circuit affordable.

## Masking with a large finite penalty

Programs exclude entries by adding a penalty instead of branching. With a mask $\mu_i \in \{0, 1\}$ and $B = 10^{30}$:

$$\tilde s_i = s_i\, \mu_i - (1 - \mu_i)\, B$$

For $\mu_i = 1$ this is exactly $s_i$, since $s_i \cdot 1 = s_i$ and $s_i - 0 = s_i$. For $\mu_i = 0$ it is $-B$ (the sign of $s_i \cdot 0$ does not matter after subtracting $B$). After scaling by $\beta = 32$, $e^{\beta(-B) - \max}$ underflows to exactly 0, so excluded entries get zero softmax weight. $B$ is finite on purpose: $\beta B = 3.2 \cdot 10^{31} < 3.4 \cdot 10^{38}$, the largest `f32`, so no infinity or NaN (not a number) appears. A min-reduction over masked entries uses $+B$ the same way.

**Selecting a value without branching.** If a condition $c \in \{0, 1\}$ chooses between a value $v$ and a default $-1$: $\;c\, v + (c - 1)$. That gives $v$ when $c = 1$ and $-1$ when $c = 0$, exactly, because $v \cdot 1 = v$ and $v \cdot 0 = \pm 0$.

## The memory-encode circuit

`model/prog/memory-encode.json`, compiled by `zerolm_wsp::circuits::key_encode`. It computes the key encoder of [memory-format.md](memory-format.md#key-encoder) from raw bytes. Input: a row $t \in \{0..255\}^L$, $L = 256$, holding the text's bytes at positions $1..\ell$ ($\ell \le L - 2$) and 0 elsewhere. The host copies the bytes there and inspects none of them. Bound model tensor: the trigram table $T$ (`core.memory_encoder`). Output rows are indexed by word rank $k = 1..W$, $W = L/2$; a text of $\ell \le L-2$ bytes has at most $W - 1$ words.

**Why fixed shapes suffice.** Map every separator byte to `#` and fold upper case. Then the trigrams of `#word#`, over all words, are exactly the length-3 windows whose middle byte is a word byte. Position 0 and every position after the text are 0 bytes, which are separators, so the first and last word of the text are padded too.

**classify** (constants: the bounds 48, 58, 65, 91, 97, 123, 128):

$$	ext{up}_i = [\![t_i \ge 65]\!] - [\![t_i \ge 91]\!], \qquad \text{word}_i = \big([\![t_i \ge 48]\!] - [\![t_i \ge 58]\!]\big) + \text{up}_i + \big([\![t_i \ge 97]\!] - [\![t_i \ge 123]\!]\big) + [\![t_i \ge 128]\!]$$

**fold** (constants: the case offset 32 and the pad byte 35, `#`):

$$m_i = \text{word}_i\,(t_i + 32\,\text{up}_i - 35) + 35$$

**window** (constants: the gather matrix $G \in \{0,1\}^{L \times 3L}$, $G_{j,\,3i+c} = [\![j = i - 1 + c]\!]$, and the selector $(1, 0, 0)$):

$$\text{win} = \mathrm{reshape}_{L \times 3}(G^\top m), \qquad H_i = T\big[\mathrm{FNV1a}(\text{win}_i) \bmod 4096\big] \quad (\texttt{hash\_rows},\ n = 3), \qquad p_i = \text{word}_{i-1}\ (p_0 = 0)$$

**segment** (constants: the prefix-sum matrix $P_{rc} = [\![r \le c]\!]$ and the ranks $\rho_k = k$, $\rho'_k = k + 1$):

$$\text{start}_i = \text{word}_i (1 - p_i), \qquad \text{id} = P^\top \text{start}, \qquad A_{ki} = \big([\![\text{id}_i \ge k]\!] - [\![\text{id}_i \ge k + 1]\!]\big)\,\text{word}_i$$

$$\tilde w = A H \in \mathbb{R}^{W \times 128}, \qquad n = \textstyle\sum_i \text{start}_i, \qquad \text{mask}_k = [\![n \ge k]\!]$$

**normalize** (no constants; `support` holds 1 and $\epsilon = 0$):

$$w_k = \frac{\tilde w_k}{\sqrt{\lVert \tilde w_k\rVert^2 + 0}}, \qquad \text{key} = \frac{\sum_k w_k}{\lVert \sum_k w_k\rVert}$$

With $\epsilon = 0$, `norm` maps a zero row to zero, so rows past the last word stay 0.

**Bit equality with the former host encoder.** $A$ is 0/1, so `matmul` (which skips zero terms exactly) adds each word's window rows in text order, starting from $+0$, exactly as the host loop did. The norm and the column sum also run in the host's order. A unit test (`encoder.rs`) compares every output bit for bit with the former host code on ASCII, mixed case, punctuation, digits, UTF-8, an emoji, and a maximal-length text.

**Blocks and ablation.** Zeroing `classify` makes every byte, including the zero padding, a word byte, so the whole input becomes one word. Zeroing `window` sends every window to the same row. Zeroing `segment` leaves no word. Each of these cuts exact recall to 0. Zeroing `fold` removes case folding and the `#` padding (separators become byte 0). Interior trigrams of a lower-case word use neither, so lower-case exact recall only falls to about 0.83. What `fold` carries is case invariance: upper-cased exact queries, answered by the intact encoder like lower-case ones, are what its ablation must remove, and that is what the memory gate checks.

## The memory-read circuit

`model/prog/memory-read.json`, compiled by `zerolm_wsp::circuits::attention_read`. Inputs: the encoder's outputs: the pooled query key $q \in \mathbb{R}^{128}$, the word rows $U \in \mathbb{R}^{n' \times 128}$, and the word mask $\lambda \in \{0,1\}^{n'}$ (for a write's same-key check, $U = q$ and $\lambda = 1$). Bound model tensors: keys $K$, key words $\mathcal{K}$, values $V$, metadata $\Pi$, and byte codes $E$ (see [memory-format.md](memory-format.md)). Every step below is one ISA instruction or a few; $[\![\cdot]\!]$ is the `ge` gate.

**select** (block constants: the status-column selector $e_\text{st}$ and the window bounds 0.5, 1.5):

$$\pi = \Pi\, e_\text{st}, \qquad \mu_i = [\![\pi_i \ge 0.5]\!] - [\![\pi_i \ge 1.5]\!], \qquad h_1 = [\![\textstyle\sum_i \mu_i \ge 1]\!], \quad h_2 = [\![\textstyle\sum_i \mu_i \ge 2]\!]$$

Status values are the exact integers 0 to 3, so $\mu_i = 1$ exactly for active slots ($\pi_i = 1$).

**attend** (constants: $\beta = 32$, $w_\min = 10^{-7}$):

$$\tilde s = (K q) \odot \mu - (1 - \mu) B, \qquad (i^\star, s^\star) = \mathrm{top}_1(\tilde s), \qquad s^{(2)} = \min \mathrm{top}_2(\tilde s)$$

$$\alpha = \mathrm{softmax}(\beta \tilde s), \qquad r = V^\top \big(\alpha \odot [\![\alpha \ge w_\min]\!]\big), \qquad \text{score} = s^\star h_1 + (h_1 - 1), \qquad \text{runner-up} = s^{(2)} h_2 + (h_2 - 1)$$

The best slot's one-hot mask $e_{i^\star}$ (from `topk`) gathers its rows by transposed `matvec`: metadata $\Pi^\top e_{i^\star}$ and key words $\mathcal{K}^\top e_{i^\star}$.

**verify** (constants: thresholds 0.3 and 0.5, the key-word-count selector, and the position index $\iota = (0, 1, \dots, 7)$):

$$n_w = (\Pi^\top e_{i^\star})_\text{kw}, \qquad \kappa = \mathrm{reshape}_{8 \times 128}(\mathcal{K}^\top e_{i^\star}), \qquad g_j = \max_i \big((\kappa U^\top)_{ji} - (1 - \lambda_i)\, B\big)$$

$$\text{cov} = h_1 \cdot [\![n_w \ge 1]\!] \cdot \min_j \big(g_j + [\![\iota_j \ge n_w]\!]\, B\big)$$

Unused word positions $j \ge n_w$ get $+B$ and drop out of the minimum; absent query rows ($\lambda_i = 0$) get $-B$ and drop out of the maximum.

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

`language-lex` (`model/prog/language-lex.json`) is the first parser circuit.
On each 128-byte chunk it classifies ASCII letters, digits, and hyphens as
word bytes, folds ASCII uppercase by adding 32, and computes word-start and
word-end masks with a position shift matrix. Descending position scores and
`topk` return the spans in input order; a sum returns their count. A tail-word
bit from the preceding chunk and the first byte's word mask determine whether
the first span continues the preceding word. The host copies the folded spans
and joins only when instructed by that bit. Padding zeroes are separators.
Input and word length are unbounded across chunks; UTF-8 bytes above 127 are
separators, matching the reference tokenizer's ASCII-only grammar.

The circuit constants contain byte ranges and position selectors, with no
vocabulary. Tests cover every byte, chunk boundaries, Unicode, and causal
ablations of classification, folding, segmentation, ordering, and continuation.
The independent interpreter replays intact and ablated circuits, and the
uniform-engine gate records lexer calls made by reasoning. Grammar
normalization and matching still run in `semantic_state::parse_tokens`.

| Circuit | Spec | Replaces | Verified by |
|---|---|---|---|
| `memory-encode` | `model/prog/memory-encode.json` (`zerolm_wsp::circuits::key_encode`) | the host key encoder: word split, case folding, `#word#` trigram hashing, normalization | bit-equality unit test against the former host code; memory gate unchanged; `reference_interpreter_parity` (every gate query); `circuit_block_ablation` per block |
| `memory-read` | `model/prog/memory-read.json` (`zerolm_wsp::circuits::attention_read`) | memory pooled attention, coverage verification, value cleanup, accept decision | memory gate unchanged; `reference_interpreter_parity` (every gate query, bit for bit); `circuit_block_ablation` per block |
| `relation-graph` (C5) | `model/prog/relation-graph.json` (`zerolm_wsp::circuits::relational`) | per-question probe, unbind, and cleanup loops over slots | reasoning gate unchanged |
| `relation-reach` (C6) | `model/prog/relation-reach.json` | breadth-first search in class inclusion, transitivity, comparison, temporal order, and the inclusion part of negation | reasoning gate unchanged, including every operator ablation through its `enable` block |

`zerolm-wsp check` recompiles every spec and compares with `model/prog/pinned.json`.

## Not yet done

- The bounded-language grammar still normalizes phrases, matches token templates,
  detects questions, and parses numerals in Rust. Mapping parsed structures to
  role bindings and dispatching by utterance form also remain host code. The
  byte tokenizer and case folder run in `language-lex`; symbol vectors run in
  `atom-encode`. Memory writes, fast-weight recall, and reasoning operator
  decisions already run in their compiled circuits.
- No circuit has been through the byte-permutation test yet.

The authoritative list is the `pending` section of `model/prog/host-audit.json`; the uniform-engine gate prints it on every run.
