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

Every symbol, role, and relation is a 512-dimensional vector of complex unit phasors (numbers $e^{i\theta}$ of modulus 1), seeded from its name. This algebra is FHRR (Fourier holographic reduced representations), one kind of VSA (vector-symbolic architecture); the math is in [Mathematics](#mathematics). The operations:

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

## Mathematics

Abbreviations: FHRR = Fourier holographic reduced representations, a vector-symbolic algebra whose vectors are complex unit phasors; VSA = vector-symbolic architecture; phasor = a complex number $e^{i\theta}$ of modulus 1. Glossary: [MATH.md](MATH.md).

### The vector algebra (`binding.rs`)

A vector is $a \in \mathbb{C}^d$, $d = 512$, stored as $[\mathrm{Re} a; \mathrm{Im} a]$ (1024 floats). An atom (symbol, role, or relation) is a unit phasor with seeded angles, $a_k = e^{i\theta_k}$, $\theta_k \sim \mathcal{U}[0, 2\pi)$, drawn from the stream tagged `binding.<namespace>:<name>`.

| Operation | Definition | Property |
|---|---|---|
| bind | $(a \circledast b)_k = a_k b_k$ | commutative, associative; a bound pair resembles neither part |
| unbind | $(a \oslash b)_k = a_k \overline{b_k}$ | $(a \circledast b) \oslash b = a$ exactly, since $\lvert b_k\rvert = 1$ |
| bundle | $a + b$ | the sum stays similar to each term |
| similarity | $\mathrm{sim}(a, b) = \frac{1}{d}\mathrm{Re}\sum_k a_k \overline{b_k} = \frac{1}{d}\sum_k \cos(\theta^a_k - \theta^b_k)$ | $\mathrm{sim}(a, a) = 1$ for an atom |

**Random atoms are quasi-orthogonal.** For independent atoms, each term $\cos(\theta^a_k - \theta^b_k)$ has mean 0 and variance $\tfrac12$, so

$$\mathbb{E}\,\mathrm{sim}(a,b) = 0, \qquad \mathrm{sd}\,\mathrm{sim}(a,b) = \frac{1}{\sqrt{2d}} = 0.031.$$

Binding an atom with anything gives another such random-looking unit phasor, so distinct role-filler pairs are quasi-orthogonal too.

### Encoding a statement

A statement with role-filler pairs $(\rho_j, f_j)$ is the bundle

$$m = \sum_{j=1}^{n} \rho_j \circledast f_j, \qquad \text{e.g. } \ \text{subj} \circledast \textit{rex} + \text{rel} \circledast \textit{isa} + \text{obj} \circledast \textit{dog}.$$

A comparative's relation filler is itself bound, $\textit{gt} \circledast \textit{height}$, and counts use $\text{rel} \circledast (\textit{has} \circledast \textit{apple})$, so each dimension or counted thing gets its own relation vector.

### Probes: thresholded matching

A probe $p = \sum_{i=1}^{k} \rho_i \circledast f_i$ (for example "subject is rex, relation is isa") is compared with every slot. If the slot contains $m$ of the probe's pairs:

$$\mathrm{sim}(p, m_\text{slot}) = m + \sum_{\text{non-matching pairs}} \mathrm{sim}(\cdot,\cdot) \approx m \pm \frac{\sqrt{kn - m}}{\sqrt{2d}}$$

The match rule is $\mathrm{sim} \ge k - \tfrac12$: all $k$ pairs present. For a 2-pair probe against 3-pair slots, the noise standard deviation is about $\sqrt{4}/32 \approx 0.06$. A full match (about 2) and a one-pair partial match (about 1) are each roughly 8 standard deviations from the threshold 1.5.

### Unbinding and cleanup

Unbinding role $\rho_r$ from a slot gives the wanted filler plus noise:

$$m \oslash \rho_r = f_r + \sum_{j \ne r} (\rho_j \circledast f_j) \oslash \rho_r$$

The noise terms are $n - 1$ quasi-orthogonal unit phasors, so $\cos(m \oslash \rho_r, f_r) \approx 1/\sqrt n$: 0.58 for 3 pairs, 0.50 for 4. **Cleanup** returns the codebook row with the highest cosine, provided it is at least 0.3. Against $N_s = 2048$ unrelated codebook rows the best spurious cosine is about $0.031\sqrt{2\ln N_s} \approx 0.12$, well under both 0.3 and the true $\ge 0.5$.

### Numbers by fractional-power encoding

With a seeded base $\phi_k \sim \mathcal{U}(-\pi, \pi)$:

$$\mathrm{num}(n)_k = e^{i n \phi_k}, \qquad \mathrm{num}(a) \circledast \mathrm{num}(b) = \mathrm{num}(a + b), \qquad \mathrm{num}(a) \oslash \mathrm{num}(b) = \mathrm{num}(a - b)$$

So addition **is** binding. For integers $a \ne b$, $\mathbb{E}\cos((a-b)\phi_k) = 0$ over the uniform base, so distinct integers are quasi-orthogonal (sd 0.031). Decoding picks $\hat n = \arg\max_{|n| \le 1000} \mathrm{sim}(v, \mathrm{num}(n))$. The scan never calls trigonometric functions per candidate: it steps a phasor, $\mathrm{num}(n+1) = \mathrm{num}(n) \circledast \mathrm{num}(1)$, in `f64`. Among 2,001 candidates the largest spurious similarity is about 0.12 against 1 for the true value.

### Discourse state ("it")

$$c_t = 0.5\, c_{t-1} + s_t \quad \Rightarrow \quad c_t = \sum_{j \ge 0} 0.5^j\, s_{t-j}$$

$s_t$ is the subject vector of statement $t$. With distinct subjects, $\lVert c\rVert^2 \approx d \sum_j 0.25^j = \tfrac43 d$, so the latest subject has cosine $1/\sqrt{4/3} = 0.87$ with $c$, the one before it 0.43, and the one before that 0.22. "it" resolves to the codebook entry with the largest cosine, if that cosine is at least 0.4; in practice this is the most recent subject. With the state ablated, $c = 0$ and nothing resolves.

### Operator semantics

Let $E^{\pm}_R$ be the positive and negative edges of relation $R$ recovered by probe, unbind, and cleanup. Let $\mathrm{reach}^D_R(x)$ be the nodes reachable from $x$ over at most $D = 8$ positive $R$ edges.

| Operator | Answers "yes" when | Answers "no" when |
|---|---|---|
| class inclusion | $y \in \mathrm{reach}^8_{isa}(x)$ | $\exists\, a \in \{x\} \cup \mathrm{reach}_{isa}(x),\ z \in \{y\} \cup \mathrm{reach}_{isa}(y)$ with $(a, z) \in E^-_{isa}$ (if $x \subseteq a$, $a \cap z = \emptyset$, and $y \subseteq z$, then $x \notin y$) |
| transitivity (in) | $y \in \mathrm{reach}^8_{in}(x)$ | $(x, y) \in E^-_{in}$ |
| attribute inheritance | at the nearest isa level $\ell$ (breadth-first from $x$) where some node has a can or cannot edge to $v$: a can edge exists | at that level a cannot edge exists |
| comparison, temporal order | $b \in \mathrm{reach}^8_{R}(a)$ | $a \in \mathrm{reach}^8_{R}(b)$ (strict order is antisymmetric) |
| conditionals | $\mathrm{prop}(x, p)$ is stated, or a rule $q \Rightarrow p$ with scope "something" or $x$ exists and $\mathrm{prop}(x, q)$ holds recursively (depth at most 8, no cycles) | $\neg\mathrm{prop}(x, p)$ is stated |
| arithmetic | $\mathrm{counts}(x) = \{n : \text{has}(x, n)\} \cup \{c + \delta : \text{more}(x, \delta, o),\ c \in \mathrm{counts}(o)\}$ has exactly one value | (more than one value is a contradiction) |

Yes and no together give **contradiction**; neither gives **unknown**. The premises are the slots on the shortest proof path, and

$$\text{confidence} = \prod_{s \in \text{premises}} \text{conf}(s), \qquad \text{conf} = 1.0 \text{ (named subject)},\ 0.9 \text{ (subject resolved from "it")}.$$

Since Phase 7, the graph search behind class inclusion, transitivity, comparison, temporal order, and the inclusion part of negation runs as the compiled circuits `relation-graph` (C5) and `relation-reach` (C6): a fixed 8-step recurrence whose premise matrix yields the shortest proof ([NEURAL_ISA.md](NEURAL_ISA.md#the-relational-circuits-phase-6-roadmap-c5-and-c6)). The remaining operator logic is still programmed Rust control flow.

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
