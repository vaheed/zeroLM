# ZeroLM

> **Zero offline pretraining. Lifelong online learning. No imported model weights.**

ZeroLM is a research project that asks a simple question: **can a neural language system start with no pretrained weights at all, and develop everything it knows from its own stream of experience?** It is written in Rust, runs on an ordinary CPU, and records every change to its weights.

Most language models are trained on huge corpora before anyone uses them. ZeroLM never is. It starts from seeded random weights plus a small number of hand-written *algorithms* (never facts or vocabulary). Everything else must be learned online, one input at a time, with each update logged and checkable.

## What works today

| Capability | What it does | Evidence | Read more |
|---|---|---|---|
| Byte-level interface | Accepts any text, including malformed UTF-8, from the first run; no tokenizer, no vocabulary | deterministic, hash-pinned initialization | [bootstrap](docs/bootstrap.md) |
| Online learning | Learns to predict the next byte from one chronological pass, with a local update rule (no backpropagation) | unseen text: 8.0 → 2.3 bits per byte; a byte-frequency baseline gets 4.2 (a byte trigram is still better at 1.2) | [online learning](docs/online-learning.md) |
| Memory in the weights | Stores taught facts as tensors in the model checkpoint and recalls them after a restart | 1,200 facts per seed: exact recall 1.000, paraphrase recall 0.973, 0.6% false answers | [memory](docs/memory-format.md) |
| Reasoning over stored facts | Answers new questions by combining stored statements: class inclusion, location, comparison, time order, arithmetic, negation, rules | 100% agreement with a symbolic reference on 3 seeds plus a hidden one; a lookup baseline gets 0% of the derived questions | [reasoning](docs/reasoning.md) |
| One neural engine | Memory retrieval and the reasoning graph search run as compiled programs over a fixed set of tensor operations | every memory query reproduced bit for bit by an independent interpreter from the exported weights | [neural ISA](docs/neural-isa.md), [weight-space programming](docs/weight-space-programming.md) |

These are small, controlled results on generated test languages, and **no claim of English understanding is made**. [What we do not claim](docs/claims-and-limitations.md) matters as much as what we do.

## Try it

With Docker (nothing else to install):

```bash
docker compose up --build                      # guided demo: reasoning, memory, online learning, with internals shown
docker compose run --rm reason                 # type statements and questions yourself
docker compose run --rm gates                  # run every check and get a pass/fail table
docker compose -f docker-compose.ghcr.yml up   # the same demo from the prebuilt image ghcr.io/vaheed/zerolm
```

With Cargo (Rust 1.75 or newer):

```console
$ printf 'every dog is a mammal\nall mammals are animals\nrex is a dog\nis rex an animal?\n' | cargo run -q -p zerolm-runtime -- reason
noted
noted
noted
yes (derived: class_inclusion; 3 premises; confidence 1.00)
```

Add `--verbose` to see each step: the parse, the tensors written, the operators and premises behind an answer, and the checkpoint hash. The [getting-started guide](docs/getting-started.md) walks through every command and shows how to check each claim yourself.

## How it works

```text
any UTF-8 text
   │
   ▼
byte interface ──► recurrent core (GRU) ──► next-byte prediction  ◄── local learning rule
                        │
                        ├──► fast weights        short-lived associations, delta-rule updates
                        ├──► fact memory         keys and values stored as checkpoint tensors
                        └──► relation memory     statements as bound phasor vectors
                                   │
                                   ▼
                     compiled circuits (attention read, graph search, ...)
                     run on one engine with a fixed set of tensor operations
```

- **Bytes in, bytes out.** Every input is a sequence of bytes 0-255. Nothing is ever out of vocabulary, and no pretrained tokenizer is needed.
- **A recurrent core.** A small gated recurrent unit (GRU) carries a fixed-size state from byte to byte. It was chosen over three other cells by a reproducible [bake-off](docs/architecture.md).
- **Learning by local rules.** Weights change only through explicit rules such as the normalized delta rule, where the change is the prediction error times the input. Every update records its input, rule, changed tensors, and the resulting checkpoint hash.
- **Memory that lives in the weights.** Facts are written into tensors of the model checkpoint and read back by an attention computation. There is no database, no transcript replay, and no hidden side file.
- **Algorithms compiled into weights.** Small algorithms are compiled into weights (weight-space programming) instead of being decided by hand-written code. They run on one engine whose fixed instruction set (the neural ISA, short for instruction set architecture) contains only tensor operations. A second, independent interpreter must reproduce every output from the weights alone.

Each doc includes the math behind its part: formulas, derivations, and why each constant works. [math.md](docs/math.md) has the notation and a glossary of every abbreviation.

## Rules we hold ourselves to

- **Learning is reported honestly.** A changed weight file is never evidence of learning; only a behavioral measurement before and after is.
- **No backpropagation in the mainline.** Local rules and compiled weights come first. A gradient-trained control may be run for comparison, and it is always labeled.
- **The model is never its own teacher.** Updates come only from the observed next byte, a user's correction, or another independently checked signal.
- **Strict data roles.** Exposure data, development tests, and hidden evaluation sets never mix.
- **Only algorithms are hand-written, never knowledge.** Compiled circuits may contain no vocabulary, word lists, facts, or corpus statistics.
- **Every gate can fail.** Thresholds are frozen before the main run, and a failed gate stops the claim.

The full rules are in the [research constitution](docs/research-constitution.md).

## Roadmap

The 1.0 goal is a genuine small language model that has learned English from a single online pass over licensed, human-written text, at roughly the level of the smallest GPT-2 on named benchmarks. Frontier-model ability is explicitly not promised: the compute gap is six or more orders of magnitude.

| Milestone | Status |
|---|---|
| Research constitution, provenance baseline, engine and architecture bake-off, byte bootstrap | done |
| Online learning, weight-resident memory, compositional reasoning | done |
| One neural engine: everything compiled into weights and verified | in progress |
| Real human-written text streams and reference baselines | planned |
| Scalable core with compiled circuits (Gate A) | planned |
| Deep representation learning without backpropagation (Gate B, a pre-registered decision) | planned |
| Scaling and forgetting (Gate C) | planned |
| Reading into memory, generation, dialogue, self-proposed weight patches | planned |
| Capability evaluation and the 1.0 release | planned |

The full plan, with every gate and kill criterion, is in [docs/roadmap.md](docs/roadmap.md). Machine-readable status: [research/milestones.json](research/milestones.json).

## Where to start reading

- **Curious about the idea:** this page, then [questions newcomers ask](docs/faq.md) (where the model is saved, why it does not grow, how much it can hold), [what we claim and do not claim](docs/claims-and-limitations.md), and [weight-space programming](docs/weight-space-programming.md).
- **Want to run it:** [getting started](docs/getting-started.md).
- **Want the details and the math:** [architecture](docs/architecture.md), [online learning](docs/online-learning.md), [memory](docs/memory-format.md), [reasoning](docs/reasoning.md), [neural ISA](docs/neural-isa.md), and [math and glossary](docs/math.md).
- **Want to contribute:** [AGENTS.md](AGENTS.md) (working rules and the checks every commit must pass), the [roadmap](docs/roadmap.md), and the [research constitution](docs/research-constitution.md).

## Repository layout

```text
crates/zerolm-tensor/     f32 kernels, deterministic random numbers and initialization
crates/zerolm-nn/         recurrent cells: GRU, diagonal state-space, fast-weight, hybrid
crates/zerolm-tokenizer/  byte tokenizer
crates/zerolm-engine/     forward pass, the neural ISA interpreter, phasor binding, bounded parser
crates/zerolm-learning/   online predictor, local rules, generated streams, n-gram baselines
crates/zerolm-memory/     weight-resident memory, fast weights, atomic checkpoints
crates/zerolm-reasoner/   relational vector memory, inference operators, symbolic reference
crates/zerolm-wsp/        circuit compiler and reversible weight patches
crates/zerolm-eval/       benchmarks and gates; the independent reference interpreter
crates/zerolm-policy/     policy audit and baseline provenance tools
crates/zerolm-runtime/    the `zerolm` command-line program (info, run, memory, reason)
model/specs/              frozen configurations and weight-group declarations
model/prog/               circuit specifications and pinned compiler-output hashes
research/                 manifests, milestone status, report schemas, stream specs
docs/                     documentation
examples/, scripts/       example inputs, demo and gate scripts for Docker
```

## Checks

Every commit must pass the tests, lint, the policy audit, and every gate:

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p zerolm-policy --bin zerolm-audit                                         # policy audit
cargo run --release -p zerolm-eval --bin zerolm-bakeoff -- --check model/specs/architecture.json
cargo run --release -p zerolm-eval --bin zerolm-online-lm                              # online learning
cargo run --release -p zerolm-eval --bin zerolm-memory                                 # memory (~7 min)
cargo run --release -p zerolm-eval --bin zerolm-reasoning                              # reasoning
cargo run -p zerolm-wsp -- check                                                       # compiled circuits
cargo run --release -p zerolm-eval --bin zerolm-uniform                                # uniform engine (~2 min)
```

## License

Apache-2.0. See [LICENSE](LICENSE).
