# Real-text data pipeline (Phase 8, in progress)

Status: the `real-text` milestone is **planned**, not complete. This page describes the part that exists: the manifest, verification, whole-document splits, and the contamination check. No corpus is fetched yet, and no baseline has been run on real text.

## Manifest

`research/corpus/manifest.json` (validated by `zerolm-data check`) lists:

- `sources`: each with `license`, `license_url`, `origin_url`, and shards (relative path, SHA-256, size). A source marked `generated_by_model` or `benchmark` is rejected, so model-generated text (for example TinyStories) and benchmark data cannot enter exposure.
- `splits`: percent of documents per role (exposure, probe, dev, hidden); they must be positive and sum to 100.
- `exposure_order`: the order in which sources are streamed (for the two-source retention test, for example public-domain books, then Simple English Wikipedia).
- `evaluation_sets`: held-out BPB (bits per byte), BLiMP, and LAMBADA files, hashed, never in exposure.

The manifest currently lists one source, `gutenberg-books`: 33 English books from a Project Gutenberg mirror (public domain in the United States), 14,524,075 bytes in one shard, one document per book. `research/corpus/fetch-gutenberg.sh` downloads them (seven listed ids were not on the mirror and are skipped); the files live in `data/`, which is git-ignored, and `zerolm-data verify --root data` checks them against the pinned SHA-256. CHILDES, Simple English Wikipedia, and English Wikipedia still need a license check before they are added.

With only 33 documents, the whole-document split leaves few books in the probe, dev, and hidden roles (hidden: 1,264,223 bytes). More sources are needed for stable held-out numbers.

## Splits

A document's role is `H(split_seed, source id, document index) mod 100` mapped through the split percentages, where `H` is FNV-1a (a 64-bit hash) followed by a bit mixer. Whole documents never straddle roles, and a document keeps its role when later documents are appended.

## Commands

```
cargo run -p zerolm-data -- check                          # manifest validation
cargo run -p zerolm-data -- verify --root DIR              # shard and evaluation hashes
cargo run -p zerolm-data -- contamination --root DIR --n 13  # n-word window overlap, exposure vs every evaluation set
```

## First baseline numbers

`cargo run --release -p zerolm-eval --bin zerolm-realtext -- --root data --exposure 1000000 --hidden 200000` (about 2.5 minutes; the learner dominates). Held-out BPB (bits per byte; 8.0 is uniform over 256 bytes) on the first 200,000 bytes of the hidden role, after one chronological pass over the first 10^6 exposure bytes:

| baseline | 10^5 bytes | 10^6 bytes |
|---|---|---|
| Witten-Bell order 1 (byte frequency) | 4.625 | 4.609 |
| Witten-Bell order 3 | 3.044 | 2.870 |
| Witten-Bell order 5 | 2.864 | 2.375 |
| online delta-rule learner (online-learning milestone) | 4.001 | 3.970 |

One seed and one split; these are measurements, not a result about ZeroLM beyond this: the Phase 4 learner beats byte frequency by about 0.64 BPB and is about 1.6 BPB behind order 5 on real text. The 10^7 and 10^8 byte points are not run (the corpus has 12.5 MB of exposure bytes and the learner needs about 20 minutes per 10^7).

## Remaining Phase 8 work

More sources, PPM and Kneser-Ney order 5, the labelled backpropagation CONTROL (a hand-written GRU backward pass), the BLiMP and LAMBADA harnesses, `model/specs/levels.json`, a report schema and CI job, and learning curves at 10^6, 10^7, and 10^8 bytes.
