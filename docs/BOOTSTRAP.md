# Bootstrap

What the system is given before it has seen any input, and what it is not.

## Language interface
- **Tokenizer** (`zerolm-tokenizer`): every UTF-8 byte is a token (IDs 0-255), plus control IDs BOS=256, EOS=257, PAD=258. Byte offsets are preserved. Any byte sequence is representable (no OOV), including malformed UTF-8; `invalid_utf8_spans` locates the bad ranges instead of rejecting input. There is no vocabulary file, merge table, or subword model. An online subword table is an experiment deferred until byte-level behaviour is stable.
- **Embedding and recurrent weights**: constructed programmatically from a seeded integer PRNG (SplitMix64), one stream per tensor tag. Nothing is downloaded. The initialization is pinned by SHA-256 in `model/bootstrap/seed.json` and checked by a test; any change must re-pin deliberately.

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
