# Concepts, relations, and compositional reasoning (Phase 6)

Phase 6 stores primitive statements as bound role-filler vectors in a tensor memory. It answers new questions by combining them with controlled inference operators. A symbolic reference reasoner labels the test questions but never runs inside the reasoner.

## Pipeline

```text
text -> semantic_state::parse        (bounded grammar -> Fact / Rule / Query structures)
     -> SemanticState                (recurrent discourse vector; resolves "it")
     -> binding                      (bundle of role (*) filler phasor bindings)
     -> memory.relations.slots       (one row per statement; polarity, confidence, provenance in meta)
question -> operators                (probe -> unbind -> cleanup -> next probe / bind numbers)
         -> Answer { verdict, derived, premises, confidence, operators }
```

### Vectors (`crates/zerolm-engine/src/binding.rs`)

Every symbol, role, and relation is a 512-dimensional complex unit phasor, seeded from its name. The operations:

- **Bind:** elementwise complex product.
- **Unbind:** product with the conjugate.
- **Bundle:** sum.

Numbers use fractional-power encoding, `number(n)_k = exp(i n θ_k)`, so `bind(number(a), number(b)) = number(a+b)`. The arithmetic operator adds by binding number vectors and then cleans the result up to an integer.

### Language (`crates/zerolm-engine/src/semantic_state.rs`)

The grammar is a small programmed one, registered as the seed entry `reasoning-grammar`:

- **Function words:** about 45.
- **Comparatives:** taller/shorter, older/younger, heavier/lighter, bigger/smaller.
- **Number words:** zero to twenty.
- **Verb normalizations:** for example "came before", "preceded", "followed", "lives in", "is unable to".
- **Pronoun:** "it".

Content words are open. Each statement type has several surface forms. For example, "every bird is an animal", "all birds are animals", and "birds are animals" all parse to the same structure, and so do "ann is shorter than tom" and "tom is taller than ann".

The discourse state updates as `c ← 0.5 c + subject` after each statement. "it" resolves to the codebook symbol nearest to `c`.

### Memory (`crates/zerolm-reasoner`)

| tensor | group | content |
|---|---|---|
| `memory.relations.slots` | `relation_memory` | 4096 x 1024: bound bundle per fact/rule, e.g. `subj(*)rex + rel(*)isa + obj(*)dog` |
| `memory.relations.meta` | `relation_memory` | polarity (+1 / -1), confidence, kind, logical time, provenance hash |
| `memory.relations.codebook` | `relation_memory` | symbol vectors seen so far (no names are stored) |
| `core.reasoner.*` | `binding_roles` | frozen role / relation phasors and the number base, verified on load |

The checkpoint uses the Phase 5 atomic format (`docs/MEMORY_FORMAT.md`). Questions and derived answers are never written to memory.

### Operators (`crates/zerolm-reasoner/src/operators.rs`)

A **probe** with k role-filler pairs matches a slot when `Re<probe, slot>/d >= k - 0.5`. **Unbinding** a role from a matched slot gives a noisy filler, which is cleaned up to the nearest codebook entry (cosine at least 0.3). Every operator below is programmed control flow over those steps. The depth limit is 8.

| operator | semantics |
|---|---|
| class_inclusion | `x isa y` if `y` is reachable over isa edges |
| transitivity | `x in y` if `y` is reachable over in edges |
| attribute_inheritance | `can(x, v)` is decided by the nearest isa level that has a can/cannot statement about `v` (specific overrides general) |
| comparison | strict order per dimension: yes if `a` reaches `b`, no if `b` reaches `a` |
| temporal_ordering | the same strict-order rule over before edges |
| arithmetic | a count is direct, or `count(other) + delta`, computed by binding number vectors |
| negation | negated statements give "no"; `no C is a D` excludes every member of `C` from `D` and its subclasses |
| conditionals | `if something is p then it is q`, and per-entity rules: modus ponens over property chains |

Both a positive and a negative proof means **contradiction**. No proof means **unknown**, so the reasoner abstains. A derived answer lists its premise slots and the operators used. Its confidence is the product of its premises' confidences: 1.0 for named statements and 0.9 for a fact whose subject was resolved from "it".

## Evaluation

The command `cargo run --release -p zerolm-eval --bin zerolm-reasoning` (about 90 s) uses the spec `model/specs/reasoning.json`.

Each seed generates a world of 221 statements:

- a 3-level taxonomy with exclusions;
- 12 abilities, each with an exception;
- 3-level containment;
- 3 comparative orders;
- 2 event chains;
- 6 counted things with relative counts;
- a rule chain;
- about 30 facts stated through "it";
- 6 injected contradictions.

All content words are pseudo-words. Each statement is taught in a random paraphrase form, and each question is asked in a different one. Labels come from the symbolic oracle. In the derived categories, a question is kept only if the lookup baseline (verbatim statements) cannot answer it, so every derived question is an unseen combination.

The hidden seed is taken from the `ZEROLM_HIDDEN_SEED` secret if it is set, otherwise from the commit hash. Only its SHA-256 hashes and aggregate scores are reported, never its seed, questions, or labels.

Main run, seeds 1-3 plus the hidden seed, about 525 questions each over 20 categories. The gates were frozen from pilot seeds 9201-9202:

| measure | result (every seed, incl. hidden) |
|---|---|
| reasoner = oracle, all questions | 1.000 |
| reasoner = oracle, derived categories | 1.000 |
| lookup baseline = oracle, derived categories | 0.000 |
| same answer for a second paraphrase of each question | 1.000 |
| counterfactual: removing one proof premise changes the answer as the oracle says | 40/40 |
| invariance: removing a statement outside the proof leaves the answer unchanged | 40/40 |
| provenance: a fresh reasoner given only the reported premises gives the same answer | 40/40 |
| answers identical after checkpoint reload | yes |
| question latency p50 / p95 | about 1.0-1.1 ms / 3.6-4.6 ms |

Ablations, as agreement on the target categories, intact → ablated:

| ablation | target categories | intact → ablated | other categories |
|---|---|---|---|
| each of the 8 operators off | its own category, e.g. arithmetic → `arithmetic_derived` | 1.000 → 0.000 | unchanged (0.000 drop) |
| negation off | excluded, exceptions, contradictions | 1.000 → 0.000 | unchanged |
| binding (roles become the identity) | all 7 derived categories | 1.000 → 0.04-0.07 | — |
| recurrent state (context stays zero) | facts stated through "it" | 1.000 → 0.000 | — |
| retrieval (no slot matches) | all answerable categories | 1.000 → 0.000 | — |

The CLI is `zerolm reason [--dir DIR]`. It stores statements and answers questions, for example `yes (derived: class_inclusion; 3 premises; confidence 1.00)`.

## What this does not show

- **No open-domain language understanding.** The language is bounded and the test content is pseudo-words. Sentences outside the grammar are rejected, not guessed.
- **The reasoning is not learned.** Each operator's control flow, and the grammar, are programmed. Only the stored statements come from input. A perfect score against the oracle is expected by construction. What the benchmark shows is that the vector pathway computes those semantics from stored primitives, and that the ablations prove this pathway is the one doing the work, rather than lookup or memorization.
- **Limited scope.**
  - Exclusion is directional: `no A is a B` does not yet imply that no B is an A.
  - There is no modus tollens.
  - Locations have no implicit disjointness.
  - "it" refers only to the most recent subject.
  - Counts derive only forward, from statements whose subject is being asked about.
- **Not language-model output.** The Phase 4 next-byte learner and the Phase 5 memory are not connected to these operators yet. Generation from this state is Phase 8.
