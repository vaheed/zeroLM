# Weight-resident memory

This component gives the model memory that persists across processes without transcript replay or an external database. Every stored fact lives only as `f32` tensors in the model checkpoint. Since the uniform-engine work, every answer is computed from those tensors by the compiled `memory-read` circuit: a program for the neural ISA (instruction set architecture, the fixed list of tensor operations the model may use; see [neural-isa.md](neural-isa.md)), whose constants are the program weights tensors `prog.memory-read.*`.

Abbreviations used below: kWTA = k-winners-take-all, FNV-1a = Fowler-Noll-Vo hash variant 1a, SHA-256 = Secure Hash Algorithm (256-bit). Glossary: [math.md](math.md).

## Checkpoint file

`DIR/memory.zlmc` uses this layout (integers little-endian):

```text
"ZLMCKPT1" | u32 header_len | header JSON | f32 tensor data ... | SHA-256 of all preceding bytes (32 bytes)
```

The header holds only the configuration (`MemoryConfig`) and the tensor directory (name, weight group, rows, cols). Facts are never written to the header. Decoding checks the magic bytes, the trailing hash, the header, and every tensor shape. It returns either the whole checkpoint or an error, never part of one.

### Atomic commit and recovery

`save` writes the file in three steps:

1. Write `memory.zlmc.tmp` and fsync it.
2. Rename the current `memory.zlmc` to `memory.prev.zlmc`.
3. Rename the temporary file to `memory.zlmc`, then fsync the directory. **This rename is the commit point.**

`load` reads `memory.zlmc`. If that file is missing or fails its hash, `load` falls back to `memory.prev.zlmc` and reports `source: previous` along with the reason. If both are invalid, it refuses to load anything. It never reads the temporary file. An uncommitted write is not a second source of truth, and there is no journal.

The tests and the benchmark inject a crash at three points: during the temporary write, before any rename, and between the two renames. After each crash, the last committed checkpoint loads with its exact hash, and the uncommitted fact is absent.

## Tensors

| tensor | group | shape (default) | role |
|---|---|---|---|
| `core.embed` | `byte_embed` | 256 x 16 | frozen byte embedding; also the code book for values |
| `core.cell.*` | `recurrent_core` | per GRU tensor | frozen GRU core |
| `core.memory_encoder` | `memory_encoder` | 4096 x 128 | frozen key encoder table |
| `core.fast_separator` | `memory_encoder` | 1024 x 128 | frozen pattern-separation projection for fast weights |
| `memory.semantic.keys` | `memory_keys` | 2048 x 128 | pooled unit key per slot |
| `memory.semantic.key_words` | `memory_keys` | 2048 x (8*128) | per-word key vectors |
| `memory.semantic.values` | `memory_values` | 2048 x (48*16) | value as embedding rows, NUL-terminated |
| `memory.semantic.meta` | `memory_meta` | 2048 x 14 | fields below |
| `memory.state` | `memory_meta` | 1 x 4 | logical clock, episode count |
| `memory.episodic.keys` | `episodic_memory` | 4096 x 128 | key of each operation (ring) |
| `memory.episodic.meta` | `episodic_memory` | 4096 x 9 | op, time, slot, source, provenance |
| `memory.fast` | `fast_weights` | 768 x 1024 | fast-weight association matrix |

Each `core.*` tensor is compared bit for bit with its pinned seeded initialization on every load, and any mismatch is rejected. Memory operations write only `memory.*`. Each event's `changed_tensors` list records this, and both the tests and the benchmark verify it.

Semantic metadata fields: `status` (0 empty, 1 active, 2 superseded, 3 deleted), `confidence` (0.5 taught, 1.0 confirmed), creation and update time, `source` (1 user statement, 2 user correction), a 64-bit provenance hash of the input stream, `supersedes`, `superseded_by`, `contradictions`, and the key's word count. Integers are split into 24-bit pieces, which `f32` represents exactly. Time is a deterministic event counter, not wall-clock time.

Semantic and episodic memory are kept separate. The semantic bank holds what is currently believed, one slot per fact. The episodic ring holds what happened, one record per operation: add, unchanged, contradiction, supersede, confirm, forget, or compact.

## Computation

- **Key encoder.** This is programmed, not learned. ASCII text is lower-cased, and any non-alphanumeric ASCII character separates words. Each word becomes the normalized sum of hashed byte-trigram rows of `core.memory_encoder`. The key is the normalized sum of the word vectors.
- **Read.** The `memory-read` circuit computes `s_i = q . K_i` over active slots, then `w = softmax(32 s)`, then `r = sum_i w_i V_i`. Each value position of `r` is cleaned up to the nearest row of `core.embed`. Equations below; the program is in [neural-isa.md](neural-isa.md#the-memory-read-circuit).
- **Accept or abstain.** The memory answers only if two conditions hold. The pooled score of the best slot must be at least 0.3. Its late-interaction coverage must be at least 0.5: for each stored key word, take the best cosine with any query word, then take the minimum over key words. Without coverage, a query that shares one of two words with a stored key cannot be told apart from a question-form paraphrase; see "Design history" below.
- **Write.** A key whose cosine with an active slot is at least 0.95 addresses that slot.
  - `teach` never overwrites a different value. It records a contradiction and returns `conflict`.
  - `correct` marks the old slot superseded, links old and new, and stores the new value.
  - `confirm` sets confidence to 1.
  - `forget` zeroes the key, key-word, and value rows of the fact and of every value it superseded. It also erases the key vectors of the matching episodic records. The metadata keeps `deleted`.
  - `compact` reclaims superseded and deleted slots and remaps links.
  - A write to a full memory is refused with `CapacityFull` and leaves memory unchanged.
- **Fast weights.** The key passes through a frozen k-winners-take-all expansion: the top 32 of 1024 units of `core.fast_separator * k`. The matrix is then updated by a normalized delta rule with a decay of 0.99 per write. fast weights holds short-lived associations only and never answers `ask`.

## Mathematics

### Key encoder (`encoder.rs`)

Text is split into words $\omega_1, \dots, \omega_n$: ASCII letters are lower-cased, and any ASCII non-alphanumeric character separates words. Each word is padded to `#word#`, and its byte trigrams $g$ select rows of the frozen table $T \in \mathbb{R}^{4096 \times 128}$, $T_{ij} \sim \mathcal{U}(-1, 1)$:

$$\tilde w(\omega) = \sum_{g \in \text{trigrams}(\#\omega\#)} T\big[\mathrm{FNV1a}(g) \bmod 4096\big], \qquad w(\omega) = \frac{\tilde w}{\lVert \tilde w\rVert}, \qquad k = \frac{\sum_{j=1}^n w(\omega_j)}{\big\lVert \sum_{j} w(\omega_j)\big\rVert}$$

**Why this separates keys.** Rows of $T$ are independent random vectors in 128 dimensions, so the word vectors of two words with no trigram in common have cosine of mean 0 and standard deviation about $1/\sqrt{128} \approx 0.09$. Treat word vectors as orthonormal. A key of $n$ words and a query of $n'$ words that share $m$ of them then have

$$\cos(k, q) \approx \frac{m}{\sqrt{n\, n'}}.$$

This explains the design history below. `zorvan size` against a stored `zorvan color` ($m = 1$, $n = n' = 2$) scores about $1/2$. The genuine paraphrase `what is the color of zorvan` ($m = 2$, $n = 2$, $n' = 6$) scores about $2/\sqrt{12} \approx 0.58$. The pooled cosine alone cannot separate the two, which is why coverage was added.

### Coverage (late interaction)

For the best slot's stored word vectors $\kappa_1..\kappa_n$ and the query's word vectors $u_1..u_{n'}$:

$$\text{cov} = \min_{j \le n}\ \max_{i \le n'}\ \langle \kappa_j, u_i \rangle$$

A key word present in the query contributes $\approx 1$. A missing one contributes the best of $n'$ near-random cosines, typically below 0.3. One missing key word is therefore enough to pull the minimum under the threshold 0.5, however many other words match.

### Attention read

Over active slots $i \in A$ with unit keys $K_i$:

$$s_i = \langle q, K_i\rangle, \qquad \alpha_i = \frac{e^{\beta s_i}}{\sum_{j \in A} e^{\beta s_j}}, \quad \beta = 32, \qquad r = \sum_{i \in A} \alpha_i\, [\![\alpha_i \ge 10^{-7}]\!]\; V_i$$

With $\beta = 32$, a runner-up whose score is $g$ below the best gets relative weight $e^{-32 g}$. That is $1.4\cdot 10^{-5}$ at the typical margin $g = 0.35$, and the $10^{-7}$ cutoff drops every slot more than $\ln(10^7)/32 \approx 0.5$ below the best. The read is soft but sharp.

### Values and cleanup

A value of bytes $v_1..v_m$ ($m \le 47$) is stored as 48 code rows, $V = [E[v_1]; \dots; E[v_m]; E[0]; \dots]$, where $E$ = `core.embed` (256 x 16). Each read position $x \in \mathbb{R}^{16}$ is decoded to the nearest code:

$$\hat b = \arg\min_b \lVert x - E_b\rVert^2 = \arg\max_b \big(2\langle x, E_b\rangle - \lVert E_b\rVert^2\big)$$

The two forms are equal because $\lVert x\rVert^2$ does not depend on $b$; the circuit uses the second, a matmul plus a bias. Decoding stops at the first $\hat b = 0$ (NUL). The **margin** is the smallest gap, over decoded positions, between the best and second-best score: how far the read is from flipping a byte.

### Accept rule

$$\text{answer} \iff s_{\text{best}} \ge 0.3 \;\wedge\; \text{cov} \ge 0.5 \;\wedge\; \hat b_1 \ne 0 \;\wedge\; |A| \ge 1$$

In the circuit this is the product of four $\{0, 1\}$ gates, multiplied into the decoded bytes, so a rejected read outputs an empty value.

**Writes** address an existing slot when $s_{\text{best}} \ge 0.95$ (same fact key). That uses the circuit's `best` and `score` outputs, queried with the write key as its only word.

### Fast weights: sparse keys and the delta rule (`fast_weights.rs`)

The pooled key $k$ is expanded and sparsified by a frozen projection $P \in \mathbb{R}^{1024 \times 128}$, $P_{ij} \sim \mathcal{U}(-1,1)$:

$$\sigma(k)_i = \frac{1}{\sqrt{a}}\, [\![\, i \in \mathrm{top}_a(P k) \,]\!], \qquad a = 32, \qquad \lVert \sigma(k) \rVert = 1$$

Two codes overlap by $\langle \sigma(k), \sigma(k')\rangle = |\mathrm{top}_a(Pk) \cap \mathrm{top}_a(Pk')| / a$. This is much smaller than $\cos(k, k')$ for correlated keys, which is the point of pattern separation. Each write with value $v$ (768 = 48 x 16 code values) applies

$$M \leftarrow \gamma M, \qquad M \leftarrow M + \eta\, \frac{(v - M\sigma)\,\sigma^\top}{\sigma^\top\sigma + \epsilon}, \qquad \gamma = 0.99,\ \eta = 1,\ \epsilon = 10^{-6}$$

Since $\sigma^\top\sigma = 1$, right after the write $M\sigma = v$ up to $\epsilon$: the newest association is stored exactly. An older association $j$ is read back as $M\sigma_j$. Its own contribution has decayed by $\gamma$ per later write, and every later write $t$ corrupted it in proportion to the code overlap $\langle\sigma_t, \sigma_j\rangle$. That is why recall falls with load (measured below).

### Integers in `f32` tensors

An `f32` has a 24-bit significand, so every integer $0 \le x < 2^{24}$ is exact. Clocks are stored as $x = \text{lo} + 2^{24}\, \text{hi}$, two exact pieces covering 48 bits. The 64-bit provenance hash is split into 22-, 22-, and 20-bit pieces. No metadata value is ever rounded.

### Logging

Every update logs `input_stream`, `signal`, `rule`, `changed_tensors`, and a hash of the written rows. Its `checkpoint_sha256` is filled in when the next save commits it.

## Results

Command: `cargo run --release -p zerolm-eval --bin zerolm-memory`. Spec: `model/specs/memory.json`; its gates were frozen from pilot seeds 9101-9102 before the main run. Main seeds 1-3 each store 1,200 generated pseudo-word facts. Each seed then runs 120 confirms, 50 contradictions, 100 supersessions, and 100 deletions, followed by a restart. Everything below is measured after the restart.

| metric | mean ± sd (seeds 1-3) |
|---|---|
| exact-key recall / precision | 1.000 / 1.000 |
| non-trivial paraphrase recall / precision | 0.973 ± 0.004 / 0.993 ± 0.001 |
| question form ("what is the color of X?") recall | 0.968-0.972 |
| one-letter typo in the entity, recall | 0.909-0.942 |
| negative false-accept rate (near-miss, unrelated, forgotten) | 0.0057 ± 0.0015 |
| answers identical before and after restart | yes (bit-identical scores) |
| core tensors changed by 1,570 memory events | none |

- **Trivial paraphrases.** Case, punctuation, and word order normalize to the same key, so they score 1.0 by construction. They are reported but not gated.
- **Forgotten keys.** Up to 2 of 100 forgotten keys still got an answer: another fact's value, through a near-miss key. The forgotten value itself was never returned, and its rows are zero.
- **Interference and capacity.** Exact recall stays 1.0 from 100 facts to a full memory of 2,048. Paraphrase recall on a fixed probe of 100 facts goes from 0.983 to 0.973, and the mean best-minus-runner-up margin falls from 0.42 to 0.35. When all 2,048 slots are full, a further write is refused.
- **Ablations.** Zeroing the value tensor, zeroing the keys, or permuting key rows among slots each drops exact recall to 0.000. The answers depend on these tensors.
- **Fast-weight recall by load.** 8 associations: 1.00. 16: 0.94-1.00. 32: 0.72-0.78. 64: 0.50-0.58. 256: about 0.13. Capacity is small, and older associations fade.
- **Latency and size** (release build, 3 seeds in parallel on a 32-core Xeon). Query p50 (median) is about 0.9 ms and p95 (95th percentile) about 1.4-1.6 ms at 1,100 facts through the compiled circuit; the earlier Rust read path took about 215 µs. Write p50 is about 300 µs before the compiled circuit (writes now also run the circuit for same-key detection). The checkpoint is 24 MB. A full save takes 0.9-1.2 s, and a load takes 0.8-1.0 s.

Curves are in `reports/memory/`: `interference-capacity.csv`, `precision-recall.csv` (a sweep over the coverage threshold), and `fast-weights.csv`. The report schema is `research/schemas/memory-report.schema.json`.

## Limitations

- The encoder is a fixed random-feature bag of words, so it ignores word order: "a likes b" and "b likes a" get the same key. It has no learned synonymy, so a query with a different word for the same thing fails. A query that contains every key word plus contradicting words is still accepted.
- Facts are generated `entity attribute => value` triples of pseudo-words, not natural-language statements. The CLI's `teach K => V` relies on the user to split key and value.
- Memory is slot-addressed. Retrieval is programmed, not learned, and its soft read is sharp, so this behaves more like a content-addressable store inside the model than like distributed memory. Superposed storage is tested only in fast weights, and fast weights capacity is small.
- Every save rewrites the full 24 MB checkpoint. That is about 1 s per mutating CLI line, which is fine for experiments but not for high write rates. Incremental checkpoints are planned for the scaling milestone.
- The online learner's readout and the engine's next-byte head are not in this checkpoint yet. The memory reads nothing from the GRU state, and the GRU is stored only so the checkpoint holds the full frozen core. Connecting memory to prediction belongs to Phases 6 and 8.

## Design history

The first pilot used only a pooled-cosine threshold of 0.55. With it, about 100% of near-miss negatives were accepted. For example, `zorvan size` was answered when only `zorvan color` was stored, because sharing one of two words gives a cosine of about 0.5-0.6, the same range as question paraphrases (about 0.58). Per-word coverage verification fixed this. In the same pilot, fast weights recalled only 3 of 8 associations: keys sharing an attribute word are correlated, so the delta-rule writes interfered. The sparse expansion fixed that. Both changes were made on pilot seeds only, before the gates were frozen and before any main-seed run.
