# Getting started

This guide shows how to run ZeroLM, watch what happens inside it, and check every claim yourself. Three ways to run it:

| Way | Needs | Best for |
|---|---|---|
| A. Docker, build locally | Docker with Compose v2 | Trying it without installing Rust; the image matches your checkout exactly |
| B. Docker, prebuilt image from GitHub Packages | Docker with Compose v2 | Fastest start; nothing is compiled |
| C. Cargo on your machine | Rust 1.75 or newer | Development |

What to expect: ZeroLM today is a research system at Level L0 (see [roadmap.md](roadmap.md) sec. 7.1). It reasons over a bounded statement language, stores facts in checkpoint tensors, and shows online learning on a small generated language. It does not understand open English. [claims-and-limitations.md](claims-and-limitations.md) lists exactly what is and is not claimed.

## A. Docker, built from your checkout

```bash
git clone https://github.com/vaheed/zeroLM.git && cd zeroLM
docker compose up --build            # first build takes about 5 minutes
```

This runs the guided demo ([scripts/docker/demo.sh](../scripts/docker/demo.sh)) with verbose mode on:

1. the model at birth (pinned initialization hash, no pretrained weights);
2. reasoning over stored statements, showing the premises behind each derived answer;
3. a new process answering from the saved checkpoint;
4. all eight reasoning operators;
5. weight-resident fact memory (teach, paraphrased ask, correct, forget, across a restart);
6. an online-learning run (about 15 s), with bits per byte before and after.

Then try it yourself:

```bash
docker compose run --rm reason        # type statements and questions; Ctrl-D quits
docker compose run --rm memory        # teach K => V, ask K, correct, confirm, forget, repeat
docker compose run --rm test          # cargo test --workspace
docker compose run --rm gates         # every gate, about 5 minutes, then a pass/fail table
docker compose run --rm gates quick   # audit, tests, and clippy only
```

To feed an example file into a service, turn off the terminal with `-T`:

```bash
docker compose run --rm -T reason < examples/reasoning-more.txt
```

## B. Docker, prebuilt image from GitHub Packages

The last job of the [CI pipeline](../.github/workflows/ci.yml), `docker-image`, runs only after every test and gate job passes. It then builds and smoke-tests the image, and on pushes to `main` publishes it as `ghcr.io/vaheed/zerolm`. Pull requests build and test the image but never publish it. It publishes `latest` and `sha-<short commit>`, and a version tag for `v*` git tags. [docker-compose.ghcr.yml](../docker-compose.ghcr.yml) has the same services but uses that image:

```bash
docker compose -f docker-compose.ghcr.yml pull
docker compose -f docker-compose.ghcr.yml up                 # guided demo
docker compose -f docker-compose.ghcr.yml run --rm reason
docker compose -f docker-compose.ghcr.yml run --rm gates
ZEROLM_TAG=sha-1a2b3c4 docker compose -f docker-compose.ghcr.yml up   # a specific commit
```

You can also use plain `docker run`:

```bash
docker run --rm -it ghcr.io/vaheed/zerolm zerolm reason --verbose
```

Both compose files use the same `zerolm-data` volume, so memory saved by one is visible to the other.

> **Maintainer note:** GitHub creates a new package as private. After the first publish, open the package settings on GitHub and set visibility to **Public**, so people can pull without logging in. Until then, `docker login ghcr.io` is required.

## C. Cargo

```bash
cargo build --release --workspace
./target/release/zerolm reason --verbose < examples/reasoning.txt
./target/release/zerolm memory --dir /tmp/zlm --verbose < examples/memory.txt
./target/release/zerolm memory --dir /tmp/zlm --verbose < examples/memory-after-restart.txt
```

The full list of commands every commit must pass is in [AGENTS.md](../AGENTS.md#commands).

## Verbose mode: seeing inside

Add `--verbose` (or `-v`), or set `ZEROLM_VERBOSE=1`. Docker services turn it on by default, except `gates`. Verbose lines go to stderr, and stdout stays exactly the replies, so scripts are not affected. To turn it off: `ZEROLM_VERBOSE=0 docker compose run --rm reason`.

Example (`zerolm reason --verbose`):

```text
[2] > rex is a dog
      parsed      {"Fact":{"subj":{"Name":"rex"},"pred":{"Isa":"dog"},"negated":false}}
      stored      slot 2 (bound role-filler vector)
      update      rule=relation_write signal=user_statement stream=cli:stdin
      changed     memory.relations.slots, memory.relations.meta, memory.relations.codebook (rows sha256 f3a50c428c0d)
      checkpoint  b2bed24bc449 saved atomically to /data/demo/reason
noted
[5] > is rex an animal?
      parsed      {"Query":{"Holds":{"subj":{"Name":"rex"},"pred":{"Isa":"animal"}}}}
      verdict     yes
      how         derived by operators: class_inclusion
      premise     slot 0 <- "every dog is a mammal"
      premise     slot 1 <- "all mammals are animals"
      premise     slot 2 <- "rex is a dog"
      confidence  1.00
      note        answers are never written back to memory
yes (derived: class_inclusion; 3 premises; confidence 1.00)
```

| Label | Meaning |
|---|---|
| `[n] > ...` | input line number and text |
| `parsed` | the structure the bounded grammar produced, or "no match" |
| `stored` / `duplicate` | which memory slot was written, or that an identical statement exists |
| `update` | the logged update: rule, signal type, input stream |
| `changed` | the tensors that changed, plus a hash of the written rows |
| `checkpoint` | hash of the checkpoint saved atomically after the change |
| `verdict` / `how` | the answer and the operators that produced it |
| `premise` | the memory slots the answer rests on. Within one session, the input line that wrote each slot is shown for reading convenience. The model itself stores only vectors, so after a restart this reads "stored before this session". |
| `retrieval` (memory) | attention score and word coverage of the best memory match |
| `core` (memory) | digest of the frozen core tensors; the final line confirms they are unchanged |

For machine-readable logs use `--trace FILE` (one JSON event per line). The interactive Docker services write traces to `reports/traces/reason.jsonl` and `reports/traces/memory.jsonl` on the host (each session overwrites its file).

## Examples

| File | Shows |
|---|---|
| [examples/reasoning.txt](../examples/reasoning.txt) | class inclusion, pronoun "it", a line outside the grammar, arithmetic |
| [examples/reasoning-after-restart.txt](../examples/reasoning-after-restart.txt) | answers from a saved checkpoint in a new process (use the same `--dir`) |
| [examples/reasoning-more.txt](../examples/reasoning-more.txt) | inheritance with exceptions, negation, comparison, temporal order, conditionals |
| [examples/memory.txt](../examples/memory.txt), [examples/memory-after-restart.txt](../examples/memory-after-restart.txt) | teach, paraphrased ask, correct, forget, across processes |
| [examples/protocol.txt](../examples/protocol.txt) | the session-ledger protocol of `zerolm run` (nothing persists) |

The test `documented_examples_give_the_documented_answers` in [crates/zerolm-runtime/tests/cli.rs](../crates/zerolm-runtime/tests/cli.rs) keeps these files and their answers in sync.

The reasoning language is bounded. It accepts sentences such as `x is a y`, `every x is a y`, `all xs are ys`, `x can v`, `x cannot v`, `x is in y`, `x is taller than y`, `x happened before y`, `x has 3 things`, `x has 2 more things than y`, and `if something is p then it is q`, plus their questions. Anything else is reported as "not understood". See [reasoning.md](reasoning.md).

## Checking the claims

`docker compose run --rm gates` (or the commands in [AGENTS.md](../AGENTS.md#commands)) runs:

| Check | Passes when | Report |
|---|---|---|
| Policy audit | no pretrained artifacts or downloads; manifests valid | stdout JSON |
| Tests and clippy | all tests pass; no warnings | stdout |
| Architecture bake-off | architecture selection reproduces `model/specs/architecture.json` | `reports/bakeoff/` |
| Online learning | held-out BPB (bits per byte) clears the frozen floor (about 2.3 BPB vs 8.0 untrained, 3 seeds) | `reports/online-learning/` |
| Memory | recall, paraphrase, and restart gates (exact recall 1.000) | `reports/memory/` |
| Reasoning | agreement with the symbolic reference, counterfactuals, ablations | `reports/reasoning/` |

Baseline provenance needs a clean git checkout, so the container reports it as **NOT RUN**. Run it on the host:

```bash
cargo run -p zerolm-policy --bin zerolm-baseline -- generate && cargo run -p zerolm-policy --bin zerolm-baseline -- check
```

Each report directory holds `summary.md` (human readable) and JSON matching the schema in `research/schemas/`.

## Troubleshooting

- **Report files are owned by root.** The container runs as root. Use `sudo chown -R "$USER" reports`, or delete them with `docker compose run --rm demo rm -rf reports/<dir>`.
- **Start from empty memory.** `docker compose down -v` removes the `zerolm-data` volume.
- **The build is slow.** The release profile uses LTO. Way B avoids building entirely.
- **`gates` reports a FAIL.** Its first failing section above the summary shows the exact command; run that command alone to reproduce.
