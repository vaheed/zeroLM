# ZeroLM — Zero-Training Language Model

**Can a neural language model do useful work without being trained?**

ZeroLM is an open-source research experiment in building a small neural text-generation system with manually engineered weights. The project explores direct network design and **Weight-Space Programming** instead of conventional pretraining or fine-tuning.

> **Current status:** ZeroLM is an intentionally limited research prototype. It recognizes a small set of English request patterns (27 topics, listed [below](#what-you-can-ask)) and emits predefined response paths through a manually specified recurrent decoder. It is not a general-purpose LLM and does not yet understand arbitrary text.

**Ways to run it:** the original C runtime (stateless), a Rust port with **ZeroLM Live** (a session with memory that survives restarts) and an **OpenAI-compatible HTTP API**, and a Docker image of the Rust build. See [INSTALL.md](INSTALL.md).

## Get started

The quickest way to build, test, and run ZeroLM is on Linux or macOS.

### 1. Get the source

```bash
git clone https://github.com/vaheed/zeroLM.git
cd zeroLM
```

### 2. Install the requirements

You need:

- A C11 compiler (`cc`, GCC, or Clang)
- GNU Make
- Python 3.10 or newer (the scripts use only the Python standard library)

Check that the tools are available:

```bash
cc --version
make --version
python3 --version
```

On macOS, install Apple's command-line developer tools when needed:

```bash
xcode-select --install
```

### 3. Build and run the full test pipeline

```bash
make pipeline
```

This single command:

1. Validates `model/spec.json` and generates `weights.zlm`.
2. Compiles the C runtime.
3. Checks that the generated weight file matches the specification.
4. Runs the C runtime smoke tests.
5. Evaluates the English regression and unknown-input test sets.
6. Audits the weight graph and decoder paths.
7. Runs a small inference benchmark.
8. Writes evaluation results to `reports/latest.json` and `reports/latest.md`.

A non-zero exit code means a build step or test failed. Read the first failing command and its output before changing the model.

### 4. Start an interactive session

```bash
./run.sh
```

Try these example prompts:

```text
Hello
Who are you?
What can you do?
Explain your architecture.
How are the weights defined?
Does ZeroLM use training?
What are its limitations?
Tell me about Weight-Space Programming.
What is the weather in Paris?
/quit
```

The final example is deliberately outside the supported scope. ZeroLM should return its `unknown` response rather than invent an answer.

To send one prompt without starting an interactive session:

```bash
./zerolm "Who are you?"
./zerolm --verbose "Who are you?"
./zerolm --info
```

On Windows, `run.bat` provides a basic launch path when GCC (for example, MinGW-w64) is installed. The documented full `make pipeline` workflow is currently aimed at Linux and macOS.

### 5. Other ways to run it

```bash
# Rust port with ZeroLM Live (memory that persists across restarts)
cd rust && cargo build --release && cd ..
rust/target/release/zerolm live --tag me

# Docker (no toolchain needed)
docker compose run --rm zerolm
```

Details, options and troubleshooting are in [INSTALL.md](INSTALL.md).

## What you can ask

Each topic is one intent with a hand-written answer. Anything else gets the `unknown` response.

| Topic | Example prompt | Topic | Example prompt |
| --- | --- | --- | --- |
| greeting | hello | install | how do I install ZeroLM |
| identity | who are you | license | what license does it use |
| capabilities | what can you do | performance | how fast is it |
| thanks | thank you | languages | what languages do you support |
| farewell | goodbye | version | which version is this |
| help | how do I use ZeroLM | live | what is zerolm live |
| architecture | explain the architecture | privacy | does it send data anywhere |
| weights | show the weight file | contribute | how can I contribute |
| no_training | does it use training | topics | what topics can you talk about |
| limitations | what are your limitations | comparison | how are you different from chatgpt |
| memory | do you have memory | planning | make a plan step by step |
| coding | write code for me | math | calculate this |
| explain | explain a concept | testing | run the test suite |
| inspiration | explain weight-space programming | | |

Phrases are tied to ZeroLM or to "it/this/you" (for example `install zerolm`, `how fast is it`) rather than bare words, so unrelated questions such as "how do I install Photoshop" still get `unknown`. The cost is that some natural rewordings also fall through to `unknown`. Matching is exact-phrase and context-blind, so false positives and misses remain possible.

## Test ZeroLM

Run the complete pipeline whenever you change code, model features, weights, or response paths:

```bash
make pipeline
```

You can also run individual checks:

| Command | What it checks |
| --- | --- |
| `./zerolm --self-test` | Classifier smoke cases and decoder paths (48 checks) |
| `python3 scripts/evaluate.py --write-report` | Labeled regression cases and unknown-input abstention |
| `python3 scripts/weight_audit.py` | Model format, intent starts, state links, and decoder paths |
| `python3 scripts/benchmark.py --iterations 500` | Repeated interactive inference on the current machine |
| `python3 scripts/inspect_weights.py` | A readable summary of the manually authored model |
| `cd rust && cargo test` | Rust: 43 unit tests, 16 end-to-end tests of `zerolm live`, 20 of `zerolm serve` |
| `python3 scripts/check_openai_sdk.py HOST:PORT [KEY]` | The official OpenAI Python SDK against a running `zerolm serve` |
| `cd rust && cargo clippy --all-targets -- -D warnings` | Rust lint |
| `scripts/docker_smoke.sh` | Builds the image; checks persistence, tags, and the session lock |
| `python3 scripts/benchmark_all.py` | C vs Rust vs ZeroLM Live; writes [reports/benchmarks.md](reports/benchmarks.md) |

### What the tests do — and do not — prove

The included tests check that known prompts select their expected intent, that the decoder emits a response, and that selected out-of-scope prompts are rejected. They also check model-file consistency and basic runtime performance.

**Passing these tests does not prove general language understanding, factual accuracy, or reasoning ability.** The current regression suite is hand-authored and finite. Add new paraphrases, ambiguous prompts, misspellings, and genuinely unseen examples. Review response meaning manually; the evaluator primarily checks intent selection and non-empty output.

The Rust port is also checked against the C runtime: CI diffs their output on every test prompt. See [docs/TESTING.md](docs/TESTING.md) for all test layers.

Benchmark numbers depend on the operating system, compiler, CPU, and system load. Compare runs on the same machine and setup. The project targets small CPU-only systems, including a two-core / 2 GB RAM environment, but users should measure resource use on their own hardware.

## How to contribute

Contributions are welcome, especially reproducible experiments that improve coverage without weakening the zero-training research boundary.

### A good first contribution

1. **Find a limitation.** Run ZeroLM and identify a prompt it handles incorrectly or does not support.
2. **Add a test first.** Add the prompt and expected intent to `tests/regression.tsv`, or add an out-of-scope prompt to `tests/unknown.tsv`. Each regression row is `intent<TAB>prompt`.
3. **Save a baseline.** Run `make pipeline`, then copy the generated report:

   ```bash
   cp reports/latest.json reports/baseline.json
   ```

4. **Make one focused change.** Edit `model/spec.json` to change manually authored input features, feature weights, or response paths. For runtime or architecture work, change the relevant C code and document the design.
5. **Run the pipeline again.**

   ```bash
   make pipeline
   ```

6. **Compare the reports.**

   ```bash
   python3 scripts/compare_reports.py reports/baseline.json reports/latest.json
   ```

7. **Review and explain the result.** Keep the change only when it fixes the target case without unacceptable regressions. Include before-and-after reports in your pull request.

### Contribution rules

- Keep the shipped model English-only.
- Do not add pretrained weights, downloaded model parameters, training datasets, training loops, fine-tuning, distillation, backpropagation, or gradient-based optimization.
- The model builder may validate and serialize the hand-authored specification; it must not learn or optimize weights.
- Do not describe repeated test runs as training or learning.
- Do not claim general language understanding based on phrase coverage or passing the included tests.
- Keep the C runtime small and dependency-light, and the Rust port standard-library only.
- The Rust one-shot output must stay identical to the C runtime's (CI diffs them).
- Live mode may keep runtime state only in a separate life file and must never modify `weights.zlm`.
- Add tests for new behavior and include unknown or ambiguous cases where relevant.
- Explain limitations, false matches, and any trade-offs in your pull request.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution checklist.

## How the model works today

The current prototype has three main stages:

1. **Input feature projection:** normalized English text is checked against manually authored phrase features. Their fixed weights contribute to intent scores.
2. **Recurrent state transitions:** the selected intent activates a start state in a sparse, manually defined recurrent state graph.
3. **Token emission:** the decoder follows that path and emits its associated tokens until an end marker is reached.

The recurrent decoder is deliberately represented in a compact form so it can run with very little memory. The current implementation is not a Transformer and does not generate unrestricted prose from a learned distribution. Read [the architecture notes](docs/ARCHITECTURE.md) for implementation details.

## ZeroLM Live

`zerolm live` (Rust port) is a long-lived session around the same model. It remembers facts you tell it (and lets unused ones fade), keeps a rolling transcript so short follow-ups work, lets you teach it phrases and rate answers, and saves all of that to a separate life file. The base weights are never modified and nothing is trained.

```text
you> remember my name is Vahid
you> what is my name?          -> my name is Vahid.
you> does it use training
you> why?                      -> (answers about training again)
you> teach yo whats up as greeting
```

`--tag NAME` gives each session its own life file (`life.NAME.zlm`), and a file lock stops two sessions, or two containers, from sharing one. `--jsonl` turns it into a line-oriented JSON service. Full details: [docs/LIVE.md](docs/LIVE.md). Measured speed and memory: [reports/benchmarks.md](reports/benchmarks.md).

## HTTP API (OpenAI-compatible)

`zerolm serve` lets other programs talk to ZeroLM. It speaks the OpenAI wire format, so OpenAI SDKs and tools work by changing the base URL, and it has a native session API with memory.

```bash
rust/target/release/zerolm serve                      # http://127.0.0.1:8080
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="your-token")
client.chat.completions.create(model="zerolm", messages=[{"role": "user", "content": "what license does it use"}])
```

Model `zerolm` is stateless (the `messages` array is the context); `zerolm-live` keeps a persistent session per `user`. Streaming, `max_tokens` and typed errors work; tools/functions are rejected because ZeroLM cannot call them. The server binds to loopback by default, requires a token of 16+ characters on any other address, and has no TLS (use a reverse proxy). Everything, including limits and what differs from OpenAI, is in [docs/API.md](docs/API.md); the contract is [rust/openapi.yaml](rust/openapi.yaml).

## Zero training means zero training

ZeroLM's research boundary is intentionally strict:

- No pretrained language model or downloaded model weights.
- No supervised or self-supervised model training.
- No fine-tuning or distillation.
- No gradient descent or backpropagation-based weight optimization.
- No hidden fitting step in the model-building pipeline.

`model/spec.json` is the human-authored source of truth. `build_model.py` validates and serializes that specification into `weights.zlm`; it does not train the network.

## Weight-Space Programming

ZeroLM is inspired by research into **Weight-Space Programming**: treating neural network parameters as structures that can be inspected, designed, and edited directly.

An important inspiration is [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/). That work explores several settings, including small networks created for specific tasks and direct editing of existing pretrained models. ZeroLM takes inspiration from the direct-weight-design direction, but starts from manually specified weights and no pretrained language model.

Read [the Weight-Space Programming research notes](docs/WEIGHT_SPACE_PROGRAMMING.md) for the relationship, distinctions, and possible experiments.

## Research directions

Areas we want to explore include:

- Better compositional input representations that go beyond exact phrase coverage.
- More expressive manually designed recurrent or other compact neural architectures.
- Weight inspection, composition, and carefully documented weight-space edits.
- Robust handling of numbers, dates, entities, and simple relations.
- Explicit abstention when the model cannot support a request.
- Repeatable evaluation with unseen prompts and independent review.
- CPU-first inference under strict memory limits.

Improvements must remain measurable and must preserve the zero-training constraint.

## Project layout

```text
main.c                  C runtime (stateless): classifier + recurrent decoder
weights.zlm             generated weight file (do not edit by hand)
model/spec.json         the hand-authored model: intents, phrase weights, responses
build_model.py          serializes spec.json -> weights.zlm (no training)
rust/                   Rust port of the runtime + ZeroLM Live (std only)
  src/                  model, memory, context, live session, lock, JSON, HTTP, API server
  openapi.yaml          the HTTP API contract (OpenAPI 3.0)
  tests/                end-to-end tests of zerolm live and zerolm serve
Dockerfile, docker-compose.yml   container image and compose service
scripts/                evaluation, audit, benchmarks, patch tool, docker smoke test
tests/                  regression.tsv and unknown.tsv prompt sets
examples/               prompts used by the documented-examples check
experiments/patches/    example weight-space patch
reports/                evaluation and benchmark results
docs/                   architecture, testing, live mode, CI, hardware, research notes
```

## Documentation

- [Install and run](INSTALL.md) — C, Rust and Docker, with troubleshooting.
- [Architecture](docs/ARCHITECTURE.md) — current network and inference path, the two runtimes, and live mode.
- [ZeroLM Live](docs/LIVE.md) — memory, tags, locking, commands, JSON-lines, limits.
- [HTTP API](docs/API.md) — OpenAI-compatible and native endpoints, auth, limits, security.
- [Using ZeroLM as an intent router](docs/ROUTER.md) — your own intents, `classify`, measuring accuracy, safe per-deployment adaptation.
- [Testing and experiments](docs/TESTING.md) — test layers, benchmarks, reports, and repeatable evaluation.
- [Continuous integration](docs/CI.md) — what each CI job checks.
- [Hardware notes](docs/HARDWARE.md) — runtime and resource targets, with measurements.
- [Weight-Space Programming](docs/WEIGHT_SPACE_PROGRAMMING.md) — research inspiration and project-specific directions.
- [Contributing](CONTRIBUTING.md) — development rules and pull request expectations.
- [Changelog](CHANGELOG.md) — project changes.

## Continuous integration

GitHub Actions runs three jobs on pushes and pull requests: the full C pipeline on Ubuntu and macOS, the Rust port (tests, clippy, and a diff of its output against the C runtime), and the Docker image. See [docs/CI.md](docs/CI.md) and the [latest workflow runs](https://github.com/vaheed/zeroLM/actions).

## License

ZeroLM is released under the [MIT License](LICENSE).

---

**ZeroLM — Engineering Neural Language Models Without Training.**
