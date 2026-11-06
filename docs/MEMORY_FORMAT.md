# Weight-resident memory (Phase 5)

Phase 5 gives the model memory that persists across processes without transcript replay or an external database. Every stored fact lives only as `f32` tensors in the model checkpoint. Every answer is computed from those tensors by the retrieval code in `crates/zerolm-memory/src/retrieval.rs`.

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
| `core.fast_separator` | `memory_encoder` | 1024 x 128 | frozen pattern-separation projection for W_FAST |
| `memory.semantic.keys` | `memory_keys` | 2048 x 128 | pooled unit key per slot |
| `memory.semantic.key_words` | `memory_keys` | 2048 x (8*128) | per-word key vectors |
| `memory.semantic.values` | `memory_values` | 2048 x (48*16) | value as embedding rows, NUL-terminated |
| `memory.semantic.meta` | `memory_meta` | 2048 x 14 | fields below |
| `memory.state` | `memory_meta` | 1 x 4 | logical clock, episode count |
| `memory.episodic.keys` | `episodic_memory` | 4096 x 128 | key of each operation (ring) |
| `memory.episodic.meta` | `episodic_memory` | 4096 x 9 | op, time, slot, source, provenance |
| `memory.fast` | `fast_weights` | 768 x 1024 | W_FAST association matrix |

Each `core.*` tensor is compared bit for bit with its pinned seeded initialization on every load, and any mismatch is rejected. Memory operations write only `memory.*`. Each event's `changed_tensors` list records this, and both the tests and the benchmark verify it.

Semantic metadata fields: `status` (0 empty, 1 active, 2 superseded, 3 deleted), `confidence` (0.5 taught, 1.0 confirmed), creation and update time, `source` (1 user statement, 2 user correction), a 64-bit provenance hash of the input stream, `supersedes`, `superseded_by`, `contradictions`, and the key's word count. Integers are split into 24-bit pieces, which `f32` represents exactly. Time is a deterministic event counter, not wall-clock time.

Semantic and episodic memory are kept separate. The semantic bank holds what is currently believed, one slot per fact. The episodic ring holds what happened, one record per operation: add, unchanged, contradiction, supersede, confirm, forget, or compact.

## Computation

- **Key encoder.** This is programmed, not learned. ASCII text is lower-cased, and any non-alphanumeric ASCII character separates words. Each word becomes the normalized sum of hashed byte-trigram rows of `core.memory_encoder`. The key is the normalized sum of the word vectors.
- **Read.** The read computes `s_i = q . K_i` over active slots, then `w = softmax(32 s)`, then `r = sum_i w_i V_i`. Each value position of `r` is cleaned up to the nearest row of `core.embed`.
- **Accept or abstain.** The memory answers only if two conditions hold. The pooled score of the best slot must be at least 0.3. Its late-interaction coverage must be at least 0.5: for each stored key word, take the best cosine with any query word, then take the minimum over key words. Without coverage, a query that shares one of two words with a stored key cannot be told apart from a question-form paraphrase; see "Design history" below.
- **Write.** A key whose cosine with an active slot is at least 0.95 addresses that slot.
  - `teach` never overwrites a different value. It records a contradiction and returns `conflict`.
  - `correct` marks the old slot superseded, links old and new, and stores the new value.
  - `confirm` sets confidence to 1.
  - `forget` zeroes the key, key-word, and value rows of the fact and of every value it superseded. It also erases the key vectors of the matching episodic records. The metadata keeps `deleted`.
  - `compact` reclaims superseded and deleted slots and remaps links.
  - A write to a full memory is refused with `CapacityFull` and leaves memory unchanged.
- **W_FAST.** The key passes through a frozen k-winners-take-all expansion: the top 32 of 1024 units of `core.fast_separator * k`. The matrix is then updated by a normalized delta rule with a decay of 0.99 per write. W_FAST holds short-lived associations only and never answers `ask`.

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
- **W_FAST recall by load.** 8 associations: 1.00. 16: 0.94-1.00. 32: 0.72-0.78. 64: 0.50-0.58. 256: about 0.13. Capacity is small, and older associations fade.
- **Latency and size** (release build, 3 seeds in parallel on a 32-core Xeon). Query p50 is about 215 µs and p95 is 320-710 µs at 1,100 facts. Write p50 is about 300 µs. The checkpoint is 24 MB. A full save takes 0.9-1.2 s, and a load takes 0.8-1.0 s.

Curves are in `reports/memory/`: `interference-capacity.csv`, `precision-recall.csv` (a sweep over the coverage threshold), and `fast-weights.csv`. The report schema is `research/schemas/phase5-report.schema.json`.

## Limitations

- The encoder is a fixed random-feature bag of words, so it ignores word order: "a likes b" and "b likes a" get the same key. It has no learned synonymy, so a query with a different word for the same thing fails. A query that contains every key word plus contradicting words is still accepted.
- Facts are generated `entity attribute => value` triples of pseudo-words, not natural-language statements. The CLI's `teach K => V` relies on the user to split key and value.
- Memory is slot-addressed. Retrieval is programmed, not learned, and its soft read is sharp, so this behaves more like a content-addressable store inside the model than like distributed memory. Superposed storage is tested only in W_FAST, and W_FAST capacity is small.
- Every save rewrites the full 24 MB checkpoint. That is about 1 s per mutating CLI line, which is fine for experiments but not for high write rates. Incremental checkpoints are Phase 11 work.
- The Phase 4 readout and the engine's next-byte head are not in this checkpoint yet. The memory reads nothing from the GRU state, and the GRU is stored only so the checkpoint holds the full frozen core. Connecting memory to prediction belongs to Phases 6 and 8.

## Design history

The first pilot used only a pooled-cosine threshold of 0.55. With it, about 100% of near-miss negatives were accepted. For example, `zorvan size` was answered when only `zorvan color` was stored, because sharing one of two words gives a cosine of about 0.5-0.6, the same range as question paraphrases (about 0.58). Per-word coverage verification fixed this. In the same pilot, W_FAST recalled only 3 of 8 associations: keys sharing an attribute word are correlated, so the delta-rule writes interfered. The sparse expansion fixed that. Both changes were made on pilot seeds only, before the gates were frozen and before any main-seed run.
