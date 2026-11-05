# Programming a language model's weights by hand: a measurement report

**Author:** vaheeD · **Model:** ZeroLM v0.2.0 · **Date:** 2026-11-05
**Everything here is reproducible from this repository** (commands at the end). Every number comes from a run of
the scripts in `scripts/`; raw outputs are in `reports/`.

## TL;DR

ZeroLM is a small language model whose weights are **written by a person, not trained**: 506 phrase→intent
weights, a 1,053-state word-by-word decoder, a 37 KB text file, no gradient descent, no pretraining, no GPU.
This report asks what that is actually good for, and measures it.

| | Question | Result |
| --- | --- | --- |
| **Read** | Can you tell *why* it answered what it did, from the weights? | Yes: every decision lists the phrases that fired. Reading them found a real bug in my own change (duplicate evidence, §1). |
| **Write** | Can a person write a working model from scratch? | Yes: the 28-intent model, and a 5-intent support router in 25 lines of JSON (§2). |
| **Fix** | Can editing weights repair a failing model, without regressions? | Held-out accuracy **40.8% → 87.7%** (dev) and **43.9% → 83.7%** (frozen test), by editing weights only (§3). |
| **Harden** | Does it survive typos? | Corrupted prompts keep their answer **84.7%** of the time, against 24.3% without the fix (§4). |
| **Scale** | Would a bigger model be better? | Accuracy still rises about 4–6 points per extra 10% of phrases; no plateau yet (§5). |
| **Verify** | Does it hold up on text it was not written for? | On the public CLINC150 benchmark: **36.3%** in-scope accuracy, **94.8%** correct refusals (§6). |

It is **not** a general chatbot. It recognises phrases you give it and answers with canned text. The honest
use is an offline, deterministic **intent router** that explains itself. The limits are in §9.

---

## What is being measured

```
text ──normalize──▶ stem + spell-snap ──▶ 506 phrase features ──▶ 28 intent scores ──▶ argmax
                                                                         │
                              below 0.75: answer "unknown" ◀────────────┤
                                                                         ▼
                                           1,053-state decoder emits the canned reply word by word
```

- **Input side:** each phrase has a hand-chosen weight for one intent. Scores add up; the highest wins; below
  0.75 the model abstains with `unknown`.
- **Output side:** a recurrent decoder over 1,053 states. Each state has one successor and one output word,
  so it behaves like a linked list; the dense-equivalent matrix would have 1,694,756 slots, all fixed.
- **Two runtimes, one file:** a C11 reference (`main.c`) and a Rust port (`rust/`). CI diffs their output.
- **Zero training:** nothing in the repository fits, optimizes or distills weights. `build_model.py` only
  serializes `model/spec.json`.

---

## 1. Read: can you tell why it answered?

**Setup**
- `zerolm classify` returns the decision, the score, and the top candidates **with the phrases that matched**.
- `scripts/weight_audit.py` checks the file for dangling states, unreachable intents and dimension limits.

Every decision is a short sum you can read. For `what are you able to do for me`:

```json
{"intent":"capabilities","score":5.80,"candidates":[
  {"intent":"capabilities","score":5.80,"matched":["what are you able to do","able to do"]},
  {"intent":"identity","score":3.20,"matched":["what are you"]}]}
```

**What reading found.** Before I fixed it, the same input scored 3.20 for both `identity` and `capabilities`, a
tie broken only by list order. Another input, `benchmrk the tests`, scored **10.60** for `testing` from four
phrases (`test`, `tests`, `testing`, `benchmark`). Three of them become the *same* stem, so one piece of evidence
was counted three times. That bug came from my own stemming change, and I found it by reading the matched
phrases, not by a test. The fix (§3) now counts the heaviest phrase of each stem once: the same input scores 5.60.

---

## 2. Write: can a person write a model from scratch?

**Setup**
- The shipped model: 28 intents, 506 phrases, written by hand in `model/spec.json`.
- A second model, from nothing: `examples/router/spec.json`, a support-desk router with 5 intents and 25 phrases,
  compiled with `python3 build_model.py --spec examples/router/spec.json --out examples/router/weights.zlm`.

**Result.** It builds in milliseconds and classifies 17 labelled lines correctly. **That 100% proves nothing about
quality**: I wrote both the phrases and the test lines. The point is the workflow: spec → serialize → measure →
adjust, with no training step. §3 shows the same workflow applied to the real model, with data written *before*
the edit.

---

## 3. Fix: can editing weights repair a failing model?

**Setup**
- Two prompt sets, written *before* touching the weights, from the intent names and answers only (not from the
  phrases): `tests/heldout.tsv` (**dev**, 130 prompts) and `tests/heldout_test.tsv` (**frozen test**, 123 prompts).
  The test set was written without looking at dev misses, and was scored only to report.
- Edits: add phrase families by hand (several wordings per intent), keep a regression suite (119 prompts) and
  a refusal suite (31 prompts) green at every step. No gradients, no search.
- The author of the model also wrote both sets, so they are **less independent than real user text**.

| Step | Phrases | Dev | Test |
| --- | ---: | ---: | ---: |
| Baseline, as shipped | 281 | 40.8% | 43.9% |
| + hand-written phrase families | 510 | 85.4% | 82.9% |
| + stemming and spell-snap (§4) | 512 | 85.4% | 82.9% |
| + same-stem phrases count once, and 7 restored phrases | 506 | **87.7%** | **83.7%** |

**How the edits went wrong, and what caught it.** Adding phrases is not free:

- One new phrase, `how do i build`, turned the refusal test "how do I build a house" into an `install` answer. The
  refusal suite caught it; the phrase was removed.
- After stemming was added, four *inflected single-word* phrases (`installing`, `licensed`, `contributing`, bare
  `weights`) started matching their base words in unrelated questions ("how do I install Photoshop", "can I use
  this license plate", "how do I lose weight"). The refusal suite and the frozen test caught these, and the phrases were replaced by
  multi-word ones.
- Open-ended "explain X" requests stay `unknown` **by design**, because the project's own refusal tests say so.
  Several of my own dev/test labels ("explain how photosynthesis works", "tell me why the sky is blue") disagree
  with that policy, so they count as misses.

**Where it still fails (frozen test, 20 of 123).** Recall is the problem, not false answers: precision is 1.00 for
most intents, but `explain` F1 is 0.33, `memory` 0.40, `help` 0.67, `limitations` 0.67. The model answers
`unknown` for rewordings it has no phrase for ("is there any persistence", "give me instructions"). Refusals are
the other side of the same gap: 12 of the 15 prompts that should be refused are (recall 0.80), but only 41% of the
times it said `unknown` were right, because 17 prompts that deserved an answer were refused.

---

## 4. Harden: does it survive typos and word forms?

**Setup**
- Two fixed rules, applied identically in C and Rust, with no new weights: (1) *light stemming* of input and
  phrases, so "tests" = "test"; (2) *spell-snap*: an unknown input word of ≥6 letters becomes the **one** phrase
  word a single edit away (swap, drop, insert, substitute) that starts with the same letter.
- `scripts/robustness.py` takes every prompt a binary already routes correctly, damages its longest word (seeded),
  and counts how many keep their answer. Compared: a build without these rules vs. the current one, same weights.

| Damage | Without | With |
| --- | ---: | ---: |
| doubled letter | 8 / 32 | 32 / 34 |
| dropped letter | 7 / 34 | 24 / 37 |
| substituted letter | 8 / 39 | 35 / 39 |
| swapped letters | 14 / 47 | 42 / 47 |
| **all** | **37 / 152 = 24.3%** | **133 / 157 = 84.7%** |

**Reading it honestly.** The damaged word is the longest word, so the "without" number is low by construction. The
two columns are over slightly different sets of prompts (each binary starts from the prompts *it* got right).
The rules do **not** improve rewordings: held-out accuracy did not move when I added them (85.4% → 85.4%).

**Side effect:** a four-letter word keeps its final `e`, so "tune" and "tuned" still differ (a unit test documents it).

---

## 5. Scale: would a bigger model be better?

**Setup**
- `scripts/scaling_experiment.py` keeps a random fraction of the 506 phrases (seeded, 10 repeats, at least one
  phrase per intent), leaving responses and decoder alone, and scores the three sets.

| Phrases kept | Count | Dev | Test | Test range | Regression |
| ---: | ---: | ---: | ---: | --- | ---: |
| 10% | 51 | 26.6% | 21.8% | 18.7–25.2% | 15.5% |
| 20% | 101 | 34.3% | 30.0% | 23.6–37.4% | 24.6% |
| 30% | 152 | 44.1% | 37.6% | 30.9–43.1% | 37.7% |
| 40% | 202 | 51.8% | 45.2% | 39.8–50.4% | 47.8% |
| 50% | 253 | 58.8% | 51.4% | 41.5–57.7% | 56.2% |
| 60% | 304 | 63.8% | 59.8% | 56.1–63.4% | 67.4% |
| 70% | 354 | 71.8% | 67.2% | 60.2–70.7% | 78.2% |
| 80% | 405 | 79.2% | 73.3% | 67.5–78.0% | 84.2% |
| 90% | 455 | 83.2% | 79.5% | 75.6–82.1% | 94.0% |
| 100% | 506 | 87.7% | 83.7% | – | 100.0% |

**Result.** Held-out accuracy climbs roughly linearly, about 4–6 points per 10% of phrases, and has **not flattened**
at 506. More phrases would still help, and the model is far from saturated.

**So would a 10 MB model be better?** It depends on what "bigger" means, and my answer is: *more phrases, yes;
more parameters of the neural kind, not by themselves.*

- In this architecture, size **is** the phrase table. 10 MB of phrase lines (at roughly 40 bytes each, my estimate) is on the order of a quarter of a
  million phrases (about 500× today), which no one can write by hand. It would have to be generated
  deterministically from existing resources (a thesaurus or paraphrase list) and then audited, which keeps the
  no-gradient rule but is no longer "written by a person".
- The curve above is for phrases written to cover *these* intents. Extra phrases from generic resources will help
  less per phrase and will raise false matches, which the refusal suite would have to police.
- Today's matcher scans every phrase for every input. At 506 phrases it does ~8,000 turns/s; scanning 500× more
  would be roughly 500× slower (an estimate, not a measurement), so it would need an index (a hash on the
  first word). The C struct would need dynamic arrays (limits today: 512 phrases, 32 intents).
- A larger model with learned weights (the usual route to 10 MB) would break the project's premise. The route
  that stays inside it is compiling hand-specified circuits (attention, copying) into weights, which is a
  different experiment: it produces small models, and its benefit would be *generalisation by structure*, not size.

I would not grow the file for its own sake. The measured next step is **more and better-chosen phrase coverage**,
with an index so the speed does not collapse, evaluated on prompts from people other than the author.

---

## 6. Verify: text it was not written for

**Setup**
- CLINC150 ("small" split, Larson et al. 2019), downloaded at run time. 7 CLINC intents map onto ZeroLM intents
  (greeting, goodbye, thank you, what can I ask you, who made you, what is your name, are you a bot). The
  remaining 1,200 queries (unsupported or out-of-scope) should be **refused**.
- Written by other people, so this is the most independent number in this report, but it covers only the 7
  intents that overlap.

| | Result |
| --- | ---: |
| In-scope accuracy (7 mapped intents) | **127 / 350 = 36.3%** (was 29.7% before this work) |
| Unsupported queries correctly refused | 568 / 600 = 94.7% |
| Out-of-scope queries correctly refused | 570 / 600 = 95.0% |

**Reading it.** The model refuses almost everything it should (95%) but answers only about a third of what it
should: the same recall gap as §3, now on someone else's wording ("who am i speaking with", "could you be a person").

---

## 7. Correctness: do the two runtimes agree, and is it tested?

| Check | Result |
| --- | --- |
| C vs. Rust, `--verbose` output on tests, examples and seeded typo variants (`scripts/parity_fuzz.py`) | **3,110 inputs, 0 differences** |
| Built-in self-test (both runtimes) | 48 / 48 |
| Regression prompts / refusal prompts | 119 / 119 and 31 / 31 |
| Rust unit tests / live-mode end-to-end / HTTP API end-to-end | 47 / 16 / 20 |
| OpenAI Python SDK against the server (`scripts/check_openai_sdk.py`) | 15 / 15 |
| OpenAPI document validity | `openapi-spec-validator` OK |
| Docker smoke test (CLI, sessions, locking, API container) | passes in CI |
| CI | 4 jobs (Linux, macOS, Rust parity, Docker), green on the last commit |

Parity is checked because the two runtimes are separate implementations of the same rules, so any rule added to
one must be added to the other; the fuzzer runs on damaged inputs because typo handling is where they could drift.

---

## 8. Speed and memory

Machine: Xeon E5-2697 v4, Linux, one run (`reports/benchmarks.md`). **This machine is shared and noisy: the same
benchmark varied by up to 2× between runs. Compare rows within a run, not across machines.**

| Measure | C | Rust |
| --- | ---: | ---: |
| One-shot process, mean (start-up + answer) | 19.9 ms | 10.4 ms |
| Steady-state turns/s, same softmax decoder | 715.8 | 778.8 (`--faithful`) |
| Steady-state turns/s, default pointer-walk decoder | – | 8,190 |
| Peak memory of a running process | 2.9 MB | 2.9 MB |
| `ZeroLM Live` turn, life file on disk (with fsync) | – | 3.7–4.4 ms |
| `ZeroLM Live` turn, life file on tmpfs | – | 0.31 ms |

**Honest notes.** One-shot latency is almost entirely process start-up. The default decoder was measured at
~20,000 turns/s before the stemming and spell-snap rules and is lower now; part of the drop is machine noise,
part is the extra per-input work, and I have not separated the two. A Live turn on disk is dominated by the
fsync that makes each save durable.

---

## 9. Negative results and limits

- **Recall is low.** Reworded questions without a matching phrase fall to `unknown`. This is the dominant error.
- **Tolerance rules do not fix wording.** Stemming and spell-snap helped typos (§4) and did nothing for
  paraphrase (§3).
- **Optimistic test sets.** The model's author wrote both held-out sets. Real user text will score lower.
- **Policy conflicts.** "Explain X" is `unknown` by the project's own tests; labels that disagree count as misses.
- **No understanding.** There is no world knowledge, reasoning, or arithmetic. Even when the `math` intent fires, the
  answer is a canned sentence; it cannot compute 17 × 23 (and today that prompt is not even recognised).
- **Matching is context-blind.** Keywords can fire in the wrong sense ("how much memory does a human brain
  have" can reach the `memory` intent).
- **Server.** The HTTP server has no TLS and no CORS; it is meant for localhost, containers, or behind a proxy.
- **Stemming is crude.** It conflates some words and will not handle irregular forms.
- **What I did not measure:** accuracy on real users' messages, multilingual input (English ASCII only), and
  anything beyond the 28 intents.

---

## 10. Reproduce it

```bash
make pipeline                                                     # build C, tests, audit, benchmark
(cd rust && cargo build --release && cargo test)                  # Rust port: 47 + 16 + 20 tests
python3 scripts/evaluate_router.py --data tests/heldout_test.tsv  # §3
python3 scripts/robustness.py --binary /path/to/older/zerolm rust/target/release/zerolm   # §4
python3 scripts/scaling_experiment.py --json reports/scaling.json # §5
python3 scripts/online_benchmark.py                               # §6 (downloads CLINC150)
python3 scripts/parity_fuzz.py --variants 10                      # §7
python3 scripts/benchmark_all.py                                  # §8
```

Raw outputs: `reports/scaling.json`, `reports/heldout_test.json`, `reports/benchmarks.{json,md}`.
Method for the held-out sets and the editing rules: [ROUTER.md](ROUTER.md) and [TESTING.md](TESTING.md).

---

## 11. Why ZeroLM does not need GGUF

GGUF is the file format that **llama.cpp, Ollama and LM Studio** use to ship the tensors of Transformer-family
models. ZeroLM does not need it, and a GGUF export would not be useful, for five reasons:

1. **There are no tensors to store.** `weights.zlm` is a plain-text file of three kinds of lines: `FEATURE`
   (a phrase, an intent and a weight), `START`, and `NODE` (a state's successor and output word). GGUF holds
   large float arrays plus tokenizer metadata; this model has neither a tensor layout nor a tokenizer.
2. **No GGUF runtime can execute it.** A GGUF file only runs on a program that implements the model's architecture.
   llama.cpp implements Llama, Qwen, Gemma and similar; it has no ZeroLM architecture. Writing one means
   re-implementing our ~680-line runtime inside it, for no new capability.
3. **Quantization has nothing to shrink.** GGUF's main benefit is compressing multi-gigabyte float weights to 4 or 8
   bits. The whole model is 37 KB, its decoder's dense-equivalent matrix (1.69 million slots) is stored as 1,053
   successor entries, and the weights are hand-chosen numbers, not learned floats.
4. **The text format is a feature.** You can read it, `git diff` it, review a weight change in a pull request,
   and rebuild it from `model/spec.json`; a binary tensor file would remove exactly that auditability, which is
   the point of programming weights by hand.
5. **Integration already exists.** Tools that expect an LLM server get ZeroLM through the **OpenAI-compatible API**
   (`zerolm serve`, `/v1/chat/completions`), which they already speak; no model-file conversion is involved.

**When this would change.** If a future version compiles hand-specified circuits into a real Transformer-shaped
network of megabytes (§5), that network would have tensors, and exporting it to GGUF so existing runtimes can
run it would then make sense. That is a different model from the one measured here.
