# ZeroLM Next - research track

**Question:** can directly programmed weights, with no training of any kind, give a small network compositional language-processing behaviour: parse novel combinations of known primitives into a structured program, reject what it does not support, and drive a separate executor?

ZeroLM Next is an experimental parser network next to the unchanged v0.2 phrase classifier (`weights.zlm`, `main.c`, `rust/src/lib.rs`). It is **not** a language model and not a general assistant; see `ARCHITECTURE.md` section 6 and `LIMITATIONS.md`.

## The supported language

```
text    := group ( joiner group )*          joiner := ", then" | "and then" | "and" | "then" | "next" | "." | ";" | ", and then"
group   := clause
         | clause "after"  gerund-clause    execution order: second, first
         | clause "before" gerund-clause    first, second
         | "after"  gerund-clause "," clause    first, second
         | "before" gerund-clause "," clause    second, first
clause  := [ "if"|"unless" "there is" NP "," ] [ "do not"|"don't"|"never" ] action [ "if"|"unless" "there is" NP ]
action  := VERB NP DIRECTION [COUNT]        (move; count 1-9: once, twice, thrice, one..nine, 1..9, "3 times", "four steps")
         | VERB NP COLOR                    (paint)
         | VERB NP                          (remove)
NP      := [the|a|an] [COLOR] NOUN          NOUN: block|cube|box, circle|disc|ball, triangle|pyramid, star
```

Semantics are defined in `ARCHITECTURE.md` and the executor: operations run in execution order on an 8×8 grid; a negated operation is skipped; a condition (`if/unless there is NP`) tests whether an object matching the type and optional colour exists *at the moment the operation runs*; a reference must match exactly one object, otherwise the operation reports `no_such_object` / `ambiguous_object`; leaving the grid reports `out_of_bounds`. A move with no count moves once.

Anything else (unknown words, pronouns, coordination of objects, ellipsis, chains of `after`/`before`, count before direction, ...) must be rejected.

## Rules of the experiment

* No pretrained weights, no datasets used to build the network, no gradients, optimizers, search or distillation. Every weight is 0, ±1, an embedding 1, or a documented constant. `research/build_next.py` derives them; `python3 research/build_next.py --check` proves `model/next/weights.zln` is exactly that output.
* Repairs are manual: `research/tools/wsp.py` diagnoses a failing trace, validates a patch written by a person, applies it with a backup, compares before/after on target and regression suites and reverts on regression. A patch records its hypothesis, affected weights, mathematical effect, expected and observed result. Patches live in `model/next/patches/` (applied) and `experiments/next-patches/` (with recorded results).
* The executor never sees text. Operations come from the network only.

## Layout

```
research/ARCHITECTURE.md  HYPOTHESES.md  RESULTS.md  LIMITATIONS.md  README.md
research/build_next.py             weight construction (spec + patches -> weights.zln)
research/benchmark/                grammar.json, scenarios.py (reference executor), generate.py (seeded), adversarial.py,
                                   defects.py, evaluate.py (metrics), run_suite.py (reports), manifest.py
research/baselines/                symbolic.py (recursive-descent parser), simple.py (keyword, phrase lookup)
research/tools/wsp.py              weight-space programming workflow
model/next/                        spec.json (lexicon), patches/, weights.zln
rust/src/next/                     encoder, network, binding, decoder, model, trace, world (executor), cli
rust/tests/next.rs                 behaviour, shortcut-detection and causal tests
tests/next/                        dev, primitives, composition sample, adversarial, defects, heldout_manifest.json
reports/next/                      baseline.json composition.json ablation.json scaling.json SUMMARY.md
experiments/next-patches/          the four manual weight patches with their recorded results
```

## Run it

```bash
make all && cargo build --release --manifest-path rust/Cargo.toml      # C reference + Rust runtime
R=rust/target/release/zerolm
$R next parse "Move the red block left twice, then move the blue circle up once."
$R next run   "paint the red block blue, then move the blue block down 2 times"   # executes on a demo world
$R next trace "move the red block up after moving the blue circle left" | python3 -m json.tool | less
$R next repl                                                                       # interactive
$R next batch cases.jsonl --summary timing.json                                    # JSON lines in, JSON lines out
$R next --ablate binding parse "..."                                               # run an ablated network
python3 research/benchmark/generate.py --profile final --seed 20260701 --dev tests/next/dev.jsonl --out tests/next/final.jsonl
python3 research/benchmark/run_suite.py            # writes reports/next/*.json and SUMMARY.md
cargo test --manifest-path rust/Cargo.toml         # includes the shortcut-detection tests
python3 research/tools/wsp.py diagnose --text "..."   # explain a failure from the trace
```

The legacy commands (`./zerolm`, `zerolm live`, `zerolm serve`, `zerolm classify`) are untouched. There is deliberately **no HTTP endpoint** for ZeroLM Next yet: the model contract (operation schema and rejection semantics) is still moving.
