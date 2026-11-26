# Real-text data pipeline (Phase 8, in progress)

Status: the `real-text` milestone is **planned**, not complete. This page describes the part that exists: the manifest, verification, whole-document splits, and the contamination check. 

## Manifest

`research/corpus/manifest.json` (validated by `zerolm-data check`) lists:

- `sources`: each with `license`, `license_url`, `origin_url`, and shards (relative path, SHA-256, size). A source marked `generated_by_model` or `benchmark` is rejected, so model-generated text (for example TinyStories) and benchmark data cannot enter exposure.
- `splits`: percent of documents per role (exposure, probe, dev, hidden); they must be positive and sum to 100.
- `exposure_order`: the order in which sources are streamed (for the two-source retention test, for example public-domain books, then Simple English Wikipedia).
- `evaluation_sets`: held-out BPB (bits per byte), BLiMP, and LAMBADA files, hashed, never in exposure.

The manifest lists two exposure sources (order: fiction, then non-fiction, for the retention test of the representation-feasibility milestone). `gutenberg-nonfiction`: 23 public-domain non-fiction books (science, philosophy, history, essays; 17,877,674 bytes), fetched by `fetch-gutenberg.sh data nonfiction`, with its own split (60/15/10/15, seed 1: the first seed giving at least 8 MB exposure, 0.6 MB probe, 0.4 MB development, and 1.2 MB held-out, chosen from document sizes only, because 23 large books leave lumpy roles under the global 80/5/5/10 split). `zerolm-realtext` streams the first source only unless given `--all-sources`, so the baseline table below is unchanged. The first source, `gutenberg-books`: 33 English books from a Project Gutenberg mirror (public domain in the United States), 14,524,075 bytes in one shard, one document per book. `research/corpus/fetch-gutenberg.sh` downloads them (seven listed ids were not on the mirror and are skipped); the files live in `data/`, which is git-ignored, and `zerolm-data verify --root data` checks them against the pinned SHA-256. CHILDES, Simple English Wikipedia, and English Wikipedia still need a license check before they are added.

With only 33 documents, the whole-document split leaves few books in the probe, dev, and hidden roles (hidden: 1,264,223 bytes). More sources are needed for stable held-out numbers.

## Simple English Wikipedia

`simple-wikipedia` (116,422 articles, 218,893,245 bytes) is the third exposure source, after the two book sources. It is built from the Wikimedia dump of 2026-11-01 by `research/corpus/fetch-simplewiki.sh` and `wiki_clean.py` (stdlib only, deterministic: main-namespace non-redirect articles; templates, references, tables, files, categories, markup, and lines under 40 characters removed; HTML entities decoded; some stray markup survives). License: CC BY-SA 4.0 (text by Wikipedia contributors, also GFDL; attribution via the article histories at simple.wikipedia.org). Wikimedia deletes old dumps, so if the URL disappears the pinned hash cannot be reproduced and the manifest must be re-pinned deliberately. Its documents use the global 80/5/5/10 split, which gives a Wikipedia held-out role of about 10% of articles for the level thresholds that are stated on Wikipedia text.

## Fresh held-out sources

`gutenberg-fresh-fiction` (5 books, 3.1 MB) and `gutenberg-fresh-nonfiction` (3 books, 0.9 MB) have `fixed_role: hidden` in the manifest: every document is held-out text, the sources are not in `exposure_order`, and no earlier run has seen them. They are the held-out role of the Gate A rerun ([representation-learning.md](representation-learning.md)); `fetch-gutenberg.sh data fresh-fiction` and `... fresh-nonfiction` download them.

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
| CONTROL: GRU trained by backpropagation through time (excluded from strict results) | 3.553 | 3.205 | 3.189 | 2.950 |

One seed and one split, with only 33 books (a few in the hidden role), so differences of a few hundredths are not reliable. Two readings: the n-gram baselines the later milestones must beat sit near 2.2 BPB at this data size, with Kneser-Ney best; and the Phase 4 learner, which beat byte frequency on a 200,000-byte prefix of the hidden role in an earlier run (3.970 against 4.609 at 10^6 bytes), is **worse than byte frequency on the whole hidden role** (4.613 against 4.479 at 10^6 bytes). The learner is a linear readout on frozen random GRU features and was tuned on a tiny synthetic language; its constants were not retuned for real text. The learner needs about 2 minutes per 10^6 bytes, so its 10^7 point is not run. 

### The backpropagation CONTROL

`zerolm-realtext ... --control` trains the same seeded GRU (embedding, all GRU weights, and a zero-initialized readout) with truncated backpropagation through time (windows of 32 bytes, Adam, gradient-norm clip 1.0) in one chronological pass: each chunk is scored with the weights from before its update, then one step is taken. The learning rate (0.003, from 0.001, 0.003, 0.01) was chosen on the dev role only (first 200,000 exposure bytes in, 200,000 dev bytes scored: 3.062, 2.954, 3.380 BPB), never on the hidden role. The analytic gradients are checked against finite differences in `crates/zerolm-eval/tests/bptt.rs`. Its running (prequential) BPB over the exposure bytes was 3.192 at 10^5, 2.564 at 10^6, 2.394 at 10^7, and 2.419 over all 12.5M.

Readings, one seed, a small hidden split: (1) the control reaches 2.95 held-out BPB, well ahead of the Phase 4 learner (4.6) and of byte frequency (4.47), but behind every order-5 n-gram baseline (best 2.17), and its held-out numbers are not monotone in exposure (3.205 to 3.189 to 2.950), so differences of a few tenths are within the noise of a few hidden books; (2) at this size (about 74 hidden units, 20,000 cell parameters) a stronger learner on this architecture has not caught a Kneser-Ney n-gram model. The Phase 9 headroom number is the gap between the local rule and this control, so this table is its reference. This control is excluded from strict results (constitution rule 2).

## Frozen level thresholds

`model/specs/levels.json` (frozen 2026-11-26, validated by `crates/zerolm-eval/src/levels.rs`) records the thresholds of section 7.1 of the roadmap: L1 is "hidden BPB at or below the better of PPM-C order 5 and Kneser-Ney order 5, re-measured at equal exposure" (with today's measured values listed for reference); L2 and L3 carry the BLiMP, LAMBADA, and Wikipedia-BPB thresholds. The L2 and L3 numbers are the roadmap's values. They are **not yet checked against their published sources** (the GPT-2 paper's small-model results, the BLiMP paper), and the Wikipedia BPB part cannot be measured until a Wikipedia source is in the manifest. BLiMP and LAMBADA are measurable now.

## BLiMP and LAMBADA harnesses

Both are evaluation-only and listed in the manifest with license and SHA-256 (`research/corpus/fetch-benchmarks.sh` fetches them into `data/eval/`; both are CC BY 4.0).

- **BLiMP** (`blimp.jsonl`: 67 paradigm files, 67,000 pairs): a model passes a pair when it gives the acceptable sentence the higher total log-probability (each sentence scored after a newline start byte; ties count as wrong). Seven pairs in the published data have identical sentences; they are kept, so no model scores 100%. Reported: pair accuracy, paradigm mean (BLiMP's own aggregate), and per-field accuracy.
- **LAMBADA** (`lambada_test.txt`: 5,153 passages): the target is the last word, punctuation removed. The file is word-tokenized and lower-cased, so a fixed rule-based detokenizer (`evalsets::detokenize`, no model) restores spacing and quotes; case cannot be restored. An item is correct when greedy byte decoding spells exactly the target and the next greedy byte ends the word. Also reported: mean bits the model gives the target's bytes.
- **Contamination:** `zerolm-data contamination` checks BLiMP by whole-sentence verbatim match (BLiMP sentences are under 13 words, so a 13-gram check would be vacuous) and LAMBADA by 13-word windows on detokenized, lower-cased text against lower-cased exposure. Result on the current corpus: 0 of 134,000 BLiMP sentences and 0 of 306,591 LAMBADA windows shared with exposure.
- **Baselines** (`zerolm-benchmarks --root data --exposure 1000000`, order-5 models after 10^6 exposure bytes): BLiMP 0.521 Witten-Bell, 0.512 PPM-C, 0.517 Kneser-Ney (chance 0.5); LAMBADA accuracy 0.000 for all three. These are far below the L2 and L3 bars, as expected for byte 5-gram models on 1 MB; the harness tests (`crates/zerolm-eval/tests/benchmarks.rs`) check the mechanics on small fixtures, not grammatical knowledge.

## Report and CI

`zerolm-realtext` writes `reports/real-text/summary.json` (schema: `research/schemas/real-text-report.schema.json`; it records the manifest, exposure, hidden-role, and levels hashes plus the L1 bar at each exposure point). The `real-text` CI job runs the unit tests, fetches the corpus and verifies every shard hash, runs the contamination check, and runs the baselines on a 10^6-byte prefix. The count-based models live in `crates/zerolm-eval/src/ngram.rs` and are unit-tested (distributions sum to 1, no future leak).

## Remaining Phase 8 work

More sources (so the hidden role is not a few books), checking the L2 and L3 thresholds against their sources, and learning curves at 10^7 and 10^8 bytes in CI or a scheduled run. The `real-text` milestone stays `planned` until these exist.
