# Architecture

## Summary

ZeroLM v0.2.0 is a manually programmed, sparse recurrent neural text generator. It has three computational parts:

1. A phrase-feature input projection selects one intent.
2. An intent initializes a recurrent hidden-state vector.
3. A recurrent transition and output projection emit a fixed word sequence one token at a time.

The phrase matcher is a fixed feature extractor. The model does not have learned token embeddings, self-attention, a Transformer block, or a general next-token language model objective.

## 1. Input feature projection

The input is normalized by folding ASCII letters to lowercase and replacing punctuation with spaces. Each feature is a manually written phrase and a numeric weight assigned to one intent.

For intent `i`, the input score is:

```text
score[i] = sum(weight[f] for each feature f matching the input and assigned to i)
```

The runtime selects the intent with the highest score. If the best score is below `0.75`, it selects the final `unknown` intent. Feature matching uses word boundaries, so a short phrase such as `hi` does not match inside an unrelated word.

This is a hand-authored phrase classifier, not a learned language understanding layer. Its score is not a calibrated probability.

## 2. Hidden-state initialization

Each intent points to a start state. The initial logits are `+8` for the selected start state and `-8` for every other state. Softmax turns those logits into the initial hidden-state distribution.

The high score makes the starting state dominant, but the implementation still uses a vector and a softmax operation.

## 3. Recurrent transition

Every state has one declared successor. The corresponding recurrent matrix row has `+8` at that successor and `-8` at all other destinations.

A direct dense matrix multiplication would be wasteful. Since every hidden vector is normalized by softmax, the recurrent logits can be simplified:

```text
logit[target] = -8 + 16 * incoming_mass[target]
```

The `-8` term is constant for every target and does not change softmax. The runtime therefore sums only the incoming probability mass for each declared edge. This is algebraically equivalent to the dense matrix for this particular matrix design and reduces each recurrent update from O(states²) to O(states).

## 4. Output projection

Each state is associated with one output word. Its output matrix has `+8` for that word and `-8` for all other vocabulary entries.

The runtime sums hidden-state mass for each unique word, multiplies it by 16, applies softmax, and chooses the highest-scoring word. It then advances the recurrent state and repeats. An `<eos>` token ends the response.

This means the model can generate a sequence through recurrent computation, but its available sequences are explicitly authored in the model specification. It does not compose arbitrary new sentences from a learned distribution.

## 5. Weight representation

`model/spec.json` is the editable source. `build_model.py` serializes it into `weights.zlm`.

The weight file contains:

- `INTENT`: an intent name and numeric ID.
- `FEATURE`: an input phrase and its manually selected intent weight.
- `START`: the first hidden state for an intent.
- `NODE`: one recurrent edge and the word emitted by that state.

Inactive values are represented implicitly as `-8`; active recurrent and output connections are `+8`. This sparse format makes the file readable and small.

The number reported as `dense-equivalent weight slots` is a logical matrix-size estimate. It must not be described as 1,645,332 independently learned parameters. The actual values are authored by hand and most are implicit defaults.

## 6. Why it is not a Transformer

A Transformer language model normally represents tokens as vectors, uses attention to mix contextual information, and predicts a distribution over the next token using parameters learned from data. ZeroLM does none of those things. It is intentionally a much smaller, manually structured recurrent experiment.

The project keeps the word “language model” as its research direction, but the current version is best described as a constrained neural text generator with a phrase-based input adapter.

## 7. Memory and runtime

The C runtime allocates a fixed-size model structure, reads the weight file once, builds the output vocabulary once, and runs one request at a time. It has no network calls, GPU dependency, external model runtime, or persistent state. The C executable can run without Python after compilation.

Persistent state is opt-in and lives only in the Rust port's `zerolm live` mode, which keeps a separate life file (decaying facts, learned phrase weights) next to an unmodified `weights.zlm`. See [LIVE.md](LIVE.md). The one-shot CLI of both runtimes remains stateless and produces identical output (checked in CI).

## 8. Two runtimes, one model file

`weights.zlm` is read by two independent implementations:

| | C (`main.c`) | Rust (`rust/`) |
| --- | --- | --- |
| Role | reference, stateless | same one-shot behaviour, plus ZeroLM Live |
| Decoder | softmax over states and vocabulary for every token | follows successor pointers (`--faithful` runs the softmax path) |
| Vocabulary build | compares each state's token with all earlier ones (O(states x vocabulary)) | hash map (O(states)) |
| Parsing | fixed buffers, `sscanf` | bounded `String`/`Vec`; dangling state or intent references are rejected at load time |
| Limits | 32 intents, 512 features, 2048 states, 1024 vocabulary | same |

The pointer walk relies on the hidden state staying (very nearly) one-hot: the start state holds
about 99.98% of the mass, and every update re-concentrates the mass on the single declared
successor, so the softmax decoder picks the token of the state the pointer names. This is a
property of this hand-built matrix design, not a general equivalence. It is checked rather than
assumed: CI diffs the output of both runtimes, in both Rust decoder modes, on every prompt in
`tests/` and `examples/`, plus `--self-test` and `--info`. Benchmarks show the cost of
the softmax path (see [TESTING.md](TESTING.md)).

## 9. ZeroLM Live layers

Live mode wraps the same classifier and decoder. It adds, in order of precedence on each turn:

1. **Commands** (`teach`, `good`/`bad`, `again`, `what did I say`, `remember`, `forget`) handled
   before classification.
2. **Memory recall**: a question opener (`what is ...`) whose key matches a stored fact.
3. **Learned reflexes**: extra phrase-to-intent weights added to the base features when
   classifying, clamped to +/-16 so the base model keeps the final say.
4. **Follow-up resolution**: a short turn (up to 4 words) with a cue word such as `why` or `yes`
   that the classifier cannot place is classified together with the last real question.
5. **The base model**: classify, abstain below 0.75, decode.

State is a **memory** (facts with a strength that decays 3% per turn and is restored by recall,
capacity-limited with weakest-first eviction), **learned reflexes** (decay 0.5% per turn), and a
byte-budgeted **transcript**. Facts and reflexes are saved to a life file after every turn
through a temp file and rename; the transcript is not saved. A kernel file lock
(`flock`) on `<life file>.lock` ensures one live session per life file, including across
containers. `weights.zlm` is never written. Details and limits: [LIVE.md](LIVE.md).
