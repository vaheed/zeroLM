# Using ZeroLM as an intent router

ZeroLM's honest strength is routing short English text to a fixed set of intents: offline,
deterministic, well under a millisecond per message (see `reports/benchmarks.md`), no GPU, no network, a few MB of memory. You write
the intents and their trigger phrases by hand; nothing is trained. It does not understand
language: it only recognises the phrases you gave it, so it fits where inputs are predictable
(support desks, command palettes, chatbot front doors) and does badly on free-form text.
The numbers so far, honestly: the shipped 28-intent model scored 41-44% on two sets of 120-130
paraphrased prompts I wrote before widening its phrases, and 83-85% after (`tests/heldout.tsv`,
`tests/heldout_test.tsv`; the test set was frozen and scored once, and I wrote both sets, so real users
would likely score lower). On the public CLINC150 benchmark, which these weights were not written for,
in-scope accuracy is about 30%. Your accuracy depends on how well your phrases cover your users'
wording. Measure it, as below.

## 1. Write a spec

`examples/router/spec.json` is a complete small example. Each intent has `features`
(`[phrase, weight]`, weight 0.1-20) and a one-line `response`. The last intent must be named
`unknown`. Scores add up: an intent wins with the highest sum of the weights of the phrases found
in the text, and the answer is `unknown` if the best sum is below 0.75. Make specific phrases
heavier ("cancel my subscription" 3.6) than generic ones ("cancel" 1.8). Limits: 32 intents,
512 features in total.

```bash
python3 build_model.py --spec my/spec.json --out my/weights.zlm        # serialise (no training)
python3 build_model.py --spec my/spec.json --out my/weights.zlm --check # CI: fail if stale
```

## 2. Classify

```bash
printf 'I was charged twice\ntell me a joke\n' | zerolm classify --model my/weights.zlm --top 3
```

One JSON line per input: `intent` (the decision), `score`, `abstained`, and `candidates` (best
real intents first, each with the phrases that fired, so every decision is explainable).

Over HTTP: `zerolm serve --model my/weights.zlm`, then `POST /v1/classify` with `{"text": "..."}`
or `{"texts": ["...", "..."]}` (up to 100). See [API.md](API.md).

## 3. Measure it

```bash
python3 scripts/evaluate_router.py --model my/weights.zlm --data labelled.tsv --min-accuracy 0.9
```

`labelled.tsv` is `intent<TAB>text` per line; texts that should be refused are labelled
`unknown`. The report gives accuracy, per-intent precision/recall/F1, the biggest confusions, the
misses, and a threshold sweep showing how accuracy and coverage trade off if you only accept
answers scoring above a cutoff (a cutoff above 0.75 can be applied by your own code from the
returned `score`).

**Use held-out data.** Texts written by someone other than the person who wrote the phrases, ideally real
user messages. The bundled `examples/router/labelled.tsv` scores 100% only because the same
author wrote both sides; it shows the tool works, not that the model is good.

## Typos and word forms

Matching tolerates small damage without any weights: input and phrases are stemmed alike ("tests" =
"test"), and an unknown word of six or more letters is snapped to the one phrase word a single edit
away with the same first letter ("benchmrk" -> "benchmark"). On 156 correctly routed prompts, each
with its longest word swapped, shortened, doubled or mistyped (seeded; `scripts/robustness.py`), the answer survives
in 85% of cases, against 24% for the same weights without this step. It does not help with
different wording ("yo", "what's the deal"), only with spelling and word forms. Rules and limits:
[ARCHITECTURE.md](ARCHITECTURE.md).

## 4. Adapting after deployment (opt-in, no training)

The base weights file is never modified. Per deployment, a session can learn reflexes in a small
overlay: `teach <phrase> as <intent>` (+8), `good` (+1), `bad` (-4), clamped and decaying (see
[LIVE.md](LIVE.md)). Over the API: `POST /v1/sessions/{tag}/teach`, then
`POST /v1/classify` with `"session": "<tag>"` to route with that overlay. Nothing is stored by
`classify`. To make a lesson permanent for everyone, add the phrase to the spec and rebuild, so
the change is reviewed and covered by tests.

## Production checklist

- Run `serve` with a token behind a TLS proxy, or use the `zerolm-api` Docker service
  (read-only, unprivileged, 256 MB). `GET /v1/health` is open for probes.
- Keep your labelled set in the repo and run `evaluate_router.py --min-accuracy` in CI; rebuild
  and `--check` the weights from the spec.
- Log low `score` or `abstained` results and route them to a human; add the phrases you learn
  from them to the spec.
- One session tag is held by one process at a time; scale out by sharding tags, or use the
  stateless `/v1/classify` (no session) behind any number of replicas.

## How the shipped model was widened without training

Method, so you can repeat it for your own spec: write held-out prompts first and commit them;
score the baseline; add phrase families to the spec by hand (several wordings per intent, not
copies of the misses); rebuild; check the existing regression and refusal tests still pass; then
score a frozen test set once. Phrases that would turn a refusal test into an answer were left out
(for example a bare "explain" or "what version"), which is why open-ended "explain X" requests
remain unanswered by design. Measured side effects: one added phrase ("how do i build") broke the
refusal test "how do I build a house" and was removed.
