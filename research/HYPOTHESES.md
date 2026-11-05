# Hypotheses, assumptions and claims

Labels: **[V]** verified by an experiment in this repository, **[A]** design assumption, **[H]** hypothesis not tested here, **[N]** negative result, **[U]** unverified.

## Primary hypothesis (H1)

> Hand-programmed recurrent weights with tensor-product binding can implement compositional interpretation of a bounded English instruction language: novel combinations of known primitives, order, quantity, negation, conditions and longer sequences are parsed correctly, and unsupported input is rejected, with no training and no phrase table.

**Falsification criteria**, fixed before the results were produced (the thresholds of the research brief): primitive interpretation ≥ 99 %, held-out compositional exact match ≥ 95 %, multi-operation execution ≥ 90 %, counterfactual consistency ≥ 95 %, unsupported-input rejection ≥ 99 %, legacy regression 100 %; and the ablation of each designed mechanism must reduce accuracy.

**Outcome:** see `RESULTS.md`. Summary: supported on the bounded grammar [V], with the caveats below.

## Secondary hypotheses

| id | statement | status |
|---|---|---|
| H2 | Behaviour depends causally on the designed weights: random, shuffled, zeroed or permuted weights fail | [V] `reports/next/baseline.json`, `ablation.json`, `cargo test every_weight_tensor_is_causally_used` |
| H3 | Binding (outer product with a pointer), pointer advance and recurrent state reads are each necessary | [V] ablations `binding`, `pointer_shift`, `state_read`, `recurrence` |
| H4 | Order information (row time) is necessary for order-sensitive instructions | [V] ablation `time_order` |
| H5 | The network can be repaired by documented, hand-reasoned weight edits found from traces, without an optimizer | [V] patches 001-004, `experiments/next-patches/` |
| H6 | The network beats a symbolic parser | [N] it does not: it ties with it. Both implement the same grammar, and this was never claimed otherwise. The point of the experiment is that the network reaches the symbolic parser's accuracy with no training and with every decision inspectable |
| H7 | The approach extends to open-vocabulary language modelling | [H] not tested; nothing here supports or refutes it |

## Design assumptions [A]

1. The lexicon (word → primitive) is authored by hand; that is the permitted "manually specified atomic vocabulary". The model has no way to learn a new word.
2. The supported language is the grammar in `research/README.md`. The benchmark generator composes sentences from `research/benchmark/grammar.json`, which is written independently of the lexicon but describes the same language. A holdout drawn from the grammar the network was designed for can show that the construction implements the grammar; it cannot show robustness to language the author did not anticipate. The hand-written adversarial suite is the only part that probes that, and it is small (72 cases).
3. Semantics of `after`/`before` (section "Order" in `README.md`), of negation (the instruction is not executed) and of conditions (existence of an object matching a type and optional colour, evaluated at execution time) are defined by the author.
4. "Held-out" here guards against the author tuning to development examples. It is not a statistical train/test split, because there is no training. Seen/unseen splits are reported anyway.

## Negative and unexpected results [N]

* The first held-out run (seed 20260601, base construction) scored 96.2 % exact match overall and **94.4 % on compositional cases, below the 95 % target**, and accepted all 300 chained order connectors. Cause analysis in `RESULTS.md`; fixed by patches 001 and 002.
* The hand-written adversarial suite found three defects the generated benchmark could not (patches 003, 004 and a tokenizer fix). A generated benchmark only contains what its author thought to generate.
* After patch 004 part of patch 001 became redundant (`tf_w`/`OPEN`); `rank_eps` and the decoder biases are never decisive. Redundancy is reported, not removed.
* The legacy v0.2 phrase classifier cannot be compared on operation accuracy because it cannot emit operations. Its numbers are reported but are not a meaningful skill comparison.
* Unsupported constructions are rejected, not interpreted, which includes some English that a person would find reasonable (`two cells to the left`, `after that`, ellipsis such as `... and the blue circle up`).

## Unverified claims [U]

* Real GitHub Actions runs: the workflow was added and every step was executed locally, but no remote run exists until the branch is pushed.
* Behaviour on non-Linux platforms, on big-endian hosts, or with a different Rust toolchain than 1.75.
