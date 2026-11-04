# ZeroLM Live

`zerolm live` runs the ZeroLM base model as a **live session**: a
long-lived session with state that survives restarts. It is implemented in the
Rust port under [`rust/`](../rust). The C runtime stays stateless.

ZeroLM Live is a runtime layer around the same hand-authored model. It is **not** a
trained model and does not change the zero-training boundary:

- `weights.zlm` is never written. A session only writes its own *life file*.
- Nothing is optimized. There are no gradients, datasets, or loss functions.
  Every change to the life file is either something the user said explicitly
  ("remember…", "teach…") or a fixed-rule nudge from feedback ("good"/"bad").
- It still has no general language understanding. It knows 18 intents, plus
  whatever the user told it.

## Life cycle

1. **Birth.** With no life file the model is a newborn: base reflexes only,
   empty memory, empty transcript.
2. **Living.** Each turn the memory ages one tick, the turn is handled, and the
   life file is saved atomically (temp file + rename).
3. **Fading.** Facts lose 3% strength per turn and are dropped below 0.05
   (about 100 unused turns). Asking about a fact adds 0.5 back (max 1.0).
   Learned reflexes fade far more slowly (0.5% per turn).
4. **Capacity.** At most `--capacity` facts (default 64) and 256 learned
   reflexes. When full, the weakest is dropped and the reply says so.

## What it can do

| You say | Effect |
| --- | --- |
| `remember <thing> is <value>` | store a fact (`are` also works) |
| `what is <thing>` / `who is` / `tell me about` / `recall` | recall it, reinforcing it |
| `forget <thing>` | delete it |
| `teach <phrase> as <intent>` | add a learned phrase→intent weight (+8) |
| `good` / `bad` | strengthen (+1) or weaken (−4) the phrases behind the last answer |
| `again` / `repeat` | repeat the last answer |
| `what did I say` | echo your last message |
| a short follow-up such as `why?` | if it is unclear alone, it is read together with the last real question |

Context is a rolling transcript with a byte budget (`--context`, default
16 KiB). The oldest turns are dropped first; the newest is always kept.

## The life file

Plain text, line oriented, like `weights.zlm`:

```
LIFE 1
CLOCK 42
MEM 0.940900 6 <key><TAB><value>
FEAT <intent> 7.920200 <phrase>
```

Learned `FEAT` weights are added to the base feature weights at classification
time and clamped to ±16, so the base model keeps the final say.

## Interfaces

```sh
cd rust && cargo build --release
./target/release/zerolm live --model ../weights.zlm --life life.zlm
./target/release/zerolm live --jsonl --life life.zlm   # daemon: one JSON object per turn
./target/release/zerolm live --tag work                # uses life.work.zlm
```

### Tags and locking

`--tag NAME` (letters, digits, `-`, `_`, up to 32 characters) gives a session
its own life file, `life.NAME.zlm`, so separate sessions never overwrite each
other. Use either `--tag` or `--life`, not both.

A running session holds `<life file>.lock`. A second process on the same life
file is refused with a message pointing to `--tag`. A lock left behind by a
crashed process is detected (Linux, via `/proc`) and replaced. On other
platforms a leftover lock must be deleted by hand.

JSON-lines input is either a plain-text line or `{"text": "..."}`. Output is
`{"reply": "...", "intent": "greeting" | null, "age": N, "memories": M}`.

## Limitations

- Recall needs a question opener ("what is…", "who is…") and an exact
  whole-word match of the stored key. There is no paraphrase handling.
- Follow-up resolution only helps short turns, and only by reusing the previous
  intent; it cannot resolve pronouns or slots.
- The transcript is not persisted; only facts and learned reflexes are.
- English and ASCII only. One live session per life file (enforced by the lock
  above); use tags for parallel sessions.
- Replies are still the canned responses of the base model.
