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

## Baseline numbers

`cargo run --release -p zerolm-eval --bin zerolm-realtext -- --root data --exposure 20000000 --hidden 1300000 --learner-limit 1000000` (about 4 minutes). Held-out BPB (bits per byte; 8.0 is uniform over 256 bytes) on the whole hidden role (1,264,223 bytes, a few books), after one chronological pass over the first N exposure bytes (the corpus has 12,532,967):

| baseline | 10^5 | 10^6 | 10^7 | all 12.5M |
|---|---|---|---|---|
| Witten-Bell order 1 (byte frequency) | 4.477 | 4.479 | 4.470 | 4.466 |
| Witten-Bell order 3 | 3.447 | 3.006 | 2.922 | 2.903 |
| Witten-Bell order 5 | 3.259 | 2.580 | 2.272 | 2.237 |
| PPM-C order 5 (with exclusion) | 3.022 | 2.444 | 2.221 | 2.194 |
| Kneser-Ney order 5 (discount 0.75) | 2.930 | 2.391 | 2.190 | **2.170** |
| online delta-rule learner (online-learning milestone) | 4.324 | 4.613 | not run | not run |

One seed and one split, with only 33 books (a few in the hidden role), so differences of a few hundredths are not reliable. Two readings: the n-gram baselines the later milestones must beat sit near 2.2 BPB at this data size, with Kneser-Ney best; and the Phase 4 learner, which beat byte frequency on a 200,000-byte prefix of the hidden role in an earlier run (3.970 against 4.609 at 10^6 bytes), is **worse than byte frequency on the whole hidden role** (4.613 against 4.479 at 10^6 bytes). The learner is a linear readout on frozen random GRU features and was tuned on a tiny synthetic language; its constants were not retuned for real text. The learner needs about 2 minutes per 10^6 bytes, so its 10^7 point is not run. The online-backpropagation CONTROL is not run.

## Remaining Phase 8 work

More sources, the labelled backpropagation CONTROL (a hand-written GRU backward pass), the BLiMP and LAMBADA harnesses, `model/specs/levels.json`, a report schema and CI job, and learning curves at 10^6, 10^7, and 10^8 bytes.
