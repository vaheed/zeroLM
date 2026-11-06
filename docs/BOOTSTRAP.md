# Bootstrap

What the system is given before it has seen any input, and what it is not.

## Language interface
- **Tokenizer** (`zerolm-tokenizer`): every UTF-8 byte is a token (IDs 0-255), plus control IDs BOS=256, EOS=257, PAD=258. Byte offsets are preserved. Any byte sequence is representable (no OOV), including malformed UTF-8; `invalid_utf8_spans` locates the bad ranges instead of rejecting input. There is no vocabulary file, merge table, or subword model. An online subword table is an experiment deferred until byte-level behaviour is stable.
- **Embedding and recurrent weights**: constructed programmatically from a seeded integer PRNG (SplitMix64), one stream per tensor tag. Nothing is downloaded. The initialization is pinned by SHA-256 in `model/bootstrap/seed.json` and checked by a test; any change must re-pin deliberately.

## Mathematics

Abbreviations: PRNG = pseudo-random number generator, OOV = out of vocabulary, UTF-8 = Unicode Transformation Format, 8-bit, SHA-256 = Secure Hash Algorithm (256-bit), FNV-1a = Fowler-Noll-Vo hash variant 1a. More in [MATH.md](MATH.md).

### Tokenization

A string is its UTF-8 bytes $b_1, \dots, b_n$, and token $t$ has id $b_t \in \{0, \dots, 255\}$. Control ids are $\text{BOS} = 256$, $\text{EOS} = 257$, $\text{PAD} = 258$. The map from bytes to ids is the identity, so every byte string, including malformed UTF-8, is representable: there is no OOV by construction. Byte offsets come straight from the positions $t$.

### SplitMix64 (`zerolm-tensor/src/lib.rs`)

All randomness is integer arithmetic modulo $2^{64}$, so every platform produces identical streams. One step from state $s$:

$$s \leftarrow s + \texttt{0x9e3779b97f4a7c15}$$

$$z \leftarrow (s \oplus (s \gg 30)) \cdot \texttt{0xbf58476d1ce4e5b9}, \quad z \leftarrow (z \oplus (z \gg 27)) \cdot \texttt{0x94d049bb133111eb}, \quad \text{output } z \oplus (z \gg 31)$$

Here $\oplus$ is XOR, $\gg$ is a logical right shift, and the products wrap modulo $2^{64}$. The additive constant is $2^{64}/\varphi$ (with $\varphi$ the golden ratio), an odd number, so the state visits all $2^{64}$ values before repeating.

**Tagged streams.** Each tensor draws from its own stream, chosen by a text tag:

$$s_0(\text{seed}, \text{tag}) = \text{seed} \oplus \operatorname{rotl}_{17}\!\big(\operatorname{FNV1a}(\text{tag})\big)$$

$$\operatorname{FNV1a}(c_1 \dots c_m):\quad h_0 = \texttt{0xcbf29ce484222325}, \quad h_j = (h_{j-1} \oplus c_j) \cdot \texttt{0x100000001b3} \bmod 2^{64}$$

Adding or reordering tensors therefore never shifts another tensor's values. Its numbers depend only on its own tag and the seed.

### From integers to weights

$$u = \frac{\lfloor z / 2^{40} \rfloor}{2^{24}} \in [0, 1), \qquad w = (2u - 1)\, a \in [-a, a)$$

The top 24 bits fill the `f32` mantissa exactly, so the conversion involves no rounding and is identical on every platform.

| Tensor | Scale $a$ | Variance $a^2/3$ |
|---|---|---|
| `embed` (256 x 16) | 1 | 1/3 |
| cell matrices (fan-in $n$) | $1/\sqrt n$ | $1/(3n)$ |
| `head` (256 x $d$) | $0.1/\sqrt d$ | $0.01/(3d)$ |
| gate biases | `linspace`: $a_0 + (a_1 - a_0)\, i/(d-1)$ | deterministic |

Phasor angles (Phase 6) use 53 bits instead: $\theta = \frac{\lfloor z / 2^{11}\rfloor}{2^{53}} \cdot 2\pi$, which fills an `f64` mantissa.

### Pinning

`init_weights_sha256` in `model/bootstrap/seed.json` is a SHA-256 hash over a header, the cell kind, and, for each tensor, its name, its length as a `u64`, and its little-endian `f32` bits. Equal hashes mean bit-identical weights. One flipped bit in any of the 43,498 parameters (embedding 4,096 + GRU 20,202 + head 18,944 + head bias 256) changes the hash, and the test then fails.

## Teaching / control protocol
One command per line, exact lowercase words:

| line | effect |
|---|---|
| `teach K => V` | note V for K (conflicts with an existing different V are refused) |
| `ask K` | recall V and whether it is confirmed |
| `correct K => V` | replace V (becomes unconfirmed) |
| `confirm K` | mark K's value confirmed |
| `forget K` | remove K |
| `repeat` | repeat the previous response |
| anything else | observed as plain text; no claim of understanding |

Each line yields one trace event (JSONL, schema `research/schemas/phase3-trace.schema.json`) with byte spans of the line, key and value, the outcome, the response source, and the weights hash.

## What this is not
- **Not learning.** The session holds `&Model`, so it cannot modify weights; every event carries the unchanged weights hash and the CLI verifies it at exit.
- **Not model memory.** `teach`/`ask` are served from a small *session-local ledger*, labelled `source: "ledger"`. It is protocol scaffolding to make the control vocabulary testable. Weight-resident memory is Phase 5; this ledger must not be cited as evidence of it.
- **Not language understanding.** The model has no learning rule yet and predicts at chance (about 8 bits/byte). It receives every byte, but nothing in the bootstrap claims it understands English.

## Seed inventory
Listed in full in `model/bootstrap/seed.json` and `research/manifest/seed-knowledge.json`: byte alphabet, control IDs, six command words, the delimiter, architecture constants (GRU, embed 16, 20,000-parameter cell budget), and init formulas.

## CLI
```bash
cargo run -p zerolm-runtime -- info
printf 'teach sky => blue\nask sky\n' | cargo run -p zerolm-runtime -- run --trace reports/phase-3/trace.jsonl
```
`run` exits non-zero if the weights hash changed.
