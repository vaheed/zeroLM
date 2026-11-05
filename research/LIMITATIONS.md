# Limitations

* **Bounded language.** Three actions (move, paint, remove), four shapes, six colours, four directions, counts 1-9, negation, existence conditions, sequence connectors, `before`/`after` groups. Every other input is rejected with an error trace, never guessed.
* **Not an LLM.** No language modelling, no next-token prediction, no open vocabulary, no world knowledge, no dialogue. See `ARCHITECTURE.md` section 6.
* **The lexicon is closed.** A new word needs a new row in `model/next/spec.json`. There is no spell correction, no stemming (plural nouns such as `blocks` are rejected), and no tolerance for typos.
* **Capacity.** 24 rows (operations) and 400 tokens per input; beyond that the input is rejected (overflow error). Tested up to 28 operations.
* **Ambiguity is resolved by fiat.** `A, if C, B` attaches the condition to `A` (suffix reading); `A after B before C` and other chains are rejected; `after that` is rejected; pronouns and ellipsis are rejected. Prefix `before X, Y` executes `Y` first.
* **No coordination or reference.** `move the red block and the blue circle left`, `move it`, `move all blocks` are unsupported by design.
* **Sticky error mass.** Once any detector fires the whole input is rejected; there is no partial parse.
* **Redundant weights.** `tf_w`/`OPEN`, `rank_eps`, the decoder biases and `Gm_b` do not change any observable behaviour (shown by the causal test). They are left in place for auditability.
* **Benchmark scope.** The generated benchmark composes the supported grammar, so a high score shows the construction matches the grammar. It says little about language outside it. All 85 primitive cases and the 72 adversarial cases were authored by the same person who designed the weights.
* **Executor is trivially simple** (8×8 grid, overlap allowed, unique-reference resolution). Passing semantic tests shows the parse is correct for that executor; it is not a claim about grounding.
* **Platform.** Developed and measured on Linux x86-64 with Rust 1.75 and GCC 13. CI was not run remotely.
