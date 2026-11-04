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
- It still has no general language understanding. It knows 27 topics (plus "unknown"), plus
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
| `good` / `bad` (also `good answer`, `wrong`, ...) | strengthen (+1) or weaken (−4) the phrases that produced the last classified answer; if none did, it says there is nothing to adjust |
| `again` / `repeat` | repeat the last answer |
| `what did I say` | echo your last message |
| a short follow-up such as `why?` or `yes` | up to 4 words, containing a cue word (`why`, `how`, `what`, `more`, `yes`, `that`, `it`, ...), and not understood on its own: it is read together with the last real question |
| `memory` topic (e.g. "does it have memory") | answered from the real state (capacity, how to use it) instead of the canned "no memory" text |

Commands are matched before classification, so any message starting with `remember `, `forget `
or `teach ` is always treated as a command. Gibberish and short unrelated phrases do not
borrow the previous answer.

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

A running session holds an exclusive OS file lock (`flock`) on
`<life file>.lock`. A second process on the same life file, **including one in
another container sharing the volume**, is refused with a message pointing to
`--tag`. The kernel releases the lock when the holder exits or is killed, so
there is no stale-lock cleanup; the `.lock` file itself stays on disk and is
harmless. (A pid-based lock cannot work across containers, because each has its
own pid namespace.) `flock` needs a local filesystem and is not reliable on some
network filesystems. On non-Unix platforms a create-new lock file is used
instead, and a leftover one must be deleted by hand.

### Docker

```sh
docker compose run --rm zerolm                    # interactive, tag "default"
docker compose run --rm zerolm live --tag work    # another session, own memory
docker compose up -d zerolm-daemon                # optional JSON-lines service
```

Life files live in the `zerolm-data` volume. The container is read-only, has no network,
and runs as an unprivileged user. See [INSTALL.md](../INSTALL.md).

### Exit codes

`0` on a normal exit (`/quit`, `/exit`, or end of input). `2` for bad arguments, an unreadable
model, a corrupt life file (left untouched), or a busy life file.

JSON-lines input is either a plain-text line or `{"text": "..."}`. Empty lines are skipped; a
JSON object without a string `text` gets `{"error": "..."}`. Output is one object per input line:
`{"reply": "...", "intent": "greeting" | null, "age": N, "memories": M}`. `intent` is `null` for
commands (`remember`, `teach`, ...); `age` counts turns across restarts.

## Limitations

- Recall needs a question opener ("what is…", "who is…") and an exact
  whole-word match of the stored key. There is no paraphrase handling.
- Follow-up resolution only helps short turns, and only by reusing the previous
  intent; it cannot resolve pronouns or slots.
- The transcript is not persisted; only facts and learned reflexes are.
- English and ASCII only. One live session per life file (enforced by the lock
  above); use tags for parallel sessions.
- Replies are still the canned responses of the base model.
- Cost: the life file is saved, with an fsync, after every turn. That is about 3.4 ms per turn on the
  benchmark host's disk against about 0.3 ms on tmpfs ([reports/benchmarks.md](../reports/benchmarks.md)),
  so very chatty sessions on slow disks pay for it. The memory itself costs nothing measurable.
- Capacity: 256 learned reflexes (weakest dropped first). `teach` only accepts intent names that exist
  in the loaded model.
