# ZeroLM Research Constitution

Status: in force from Phase 0. Machine-checked by `zerolm-audit` against
`research/manifest/*.json` and `model/specs/weight-groups.json`.

**Definition.** ZeroLM means *zero offline pretraining; online learning is allowed.*
The earlier "zero-training / no learning" wording belonged to the removed v0.2
and ZeroLM Next experiments and no longer applies.

## Terms

| Term | Meaning |
|---|---|
| Seed prior | Human-authored structure or constant present before any input. Listed in `seed-knowledge.json`; never derived from a corpus. |
| Online observation | One event of the chronological input stream, processed once, in order. |
| Feedback | A signal tied to an observation: the observed next byte, a user correction, a deterministic environment outcome, or an independently validated signal. |
| Memory write | Store of a key/value association into persistent memory tensors. |
| Local plasticity update | Bounded local rule (e.g. normalized delta rule) changing fast weights. No global gradient. |
| WSP patch | Versioned, attributable, reversible weight edit, validated against frozen regression suites before acceptance. |
| Training | Any parameter change from a corpus outside the online stream. Forbidden in the release path. |

## Invariants

1. No pretrained weights, embeddings, tokenizer merges, or hidden checkpoints.
2. No offline pretraining or fine-tuning in the release path.
3. Every update records `input_stream`, `signal`, `rule`, `changed_tensors`, `checkpoint_sha256`.
4. **Online backpropagation is forbidden in the mainline.** An isolated, labelled control run may use it; it is excluded from strict results.
5. Exposure, development, and hidden evaluation streams are separate. Hidden labels never reach the update controller. A stream used for online exposure is training input.
6. The model's own output is never a teacher.
7. Every accepted patch is versioned, attributable, reversible, and re-evaluated.
8. Self-modification never executes code, alters the runtime binary, CI, or protected evaluation data.
9. "Seven-year-old" / "twenty-year-old" are aspiration labels; report concrete capabilities.

## Bootstrap
<a name="bootstrap"></a>
Initial language: raw UTF-8 bytes (256 symbols). Permitted knowledge inventory is exactly the entries of `research/manifest/seed-knowledge.json`.

## Initialization
<a name="initialization"></a>
Weights start from `zeros`, a documented seeded PRNG, or a named seed entry. Every weight group in `model/specs/weight-groups.json` declares its `init_source` and `update_path`.

## Learning rules
<a name="learning-rules"></a>
Mainline learning is local plasticity (normalized delta rule), memory writes, and WSP patches. Constants are fixed in Phase 4 and recorded as seed entries.

## CPU profiles
Reported per run: CPU model, cores, RAM, OS, Rust version; p50/p95 latency, tokens/s, peak RSS, checkpoint size, update cost. Targets are set in Phase 2.
