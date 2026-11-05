# Weight-Space Programming: inspiration and ZeroLM experiments

## The research inspiration

ZeroLM is inspired in part by the idea described by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab in the November 2, 2026 post:

**[Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/)**

The post describes a capable external LLM reasoning about neural-network weights directly. Instead of using gradient descent, it reads weight tensors, reasons about the computation they encode, and writes targeted weight changes or a new set of weights. The authors call this Weight-Space Programming.

The post reports several kinds of experiments:

- **MinAtar games:** The authors asked an external model to write small networks for Breakout, Seaquest, and Asterix from game rules and an input/output description. The network shape was `600 -> 32 -> 32 -> 6`; weights started at zero. In this experiment there was no training data, loss, gradient, tool use, or environment rollout before the weights were scored. The generated networks beat the reported 100-million-step PPO baselines in all three games, but performance against stronger 500-million-step baselines varied by game.
- **Go:** The authors used a roughly 1.48-million-weight network for 9x9 Go. A Python sandbox let the external model write and inspect weight tensors. They tested different budgets of forward passes; with 50 forward-pass rounds, the authored network beat an equal-sized PPO opponent trained for 20 million games. This is a structured game task with explicit rules and a defined board-state input, not free-form language generation.
- **Qwen3.5-0.8B:** The authors edited a model that was already pretrained. They identified a repeated-reasoning-loop failure and changed 131,072 weights, about 0.016% of the model. With 50 forward-pass rounds, the post reports MATH-500 rising from 7.9% to 48.1%, GSM8K from 13.8% to 67.1%, and IFEval from 21.0% to 62.3%. This was weight editing without gradient descent, not creating a language model from zero.
- **A negative result:** In an additional Gemma 3 270M experiment, the authors could not produce a clear math improvement. They note that weight editing may remove an obstacle that hides an existing capability, but it may not create a capability the model does not have.

The post is an inspiration and a research report, not a guarantee. ZeroLM does not reproduce those results and does not claim that its small handwritten model can match them.

## The important distinction

There are three different claims that should not be mixed together:

1. **No gradient training:** weights are changed without gradient descent. A capable programmer may still inspect outputs, use test feedback, and edit weights by hand.
2. **No data-driven training:** weights are written directly from a task description and a known architecture. This can work for small, structured tasks with clear inputs and outputs.
3. **General language ability from nothing:** arbitrary language understanding and broad world knowledge appear in a network with no training and no external encoding. The cited work does not establish this.

ZeroLM currently follows the second approach only for a small, bounded task: recognize authored English phrase patterns and emit one of a limited set of response paths.

## How to adapt the idea to ZeroLM

The useful concept is to let an external weight programmer inspect and edit a small, transparent model, while the repository provides strict tests and a rollback path.

The roles are separate:

- **ZeroLM:** the tiny target network. It runs offline on a basic CPU and does not update itself.
- **Weight programmer:** a human or an external capable LLM that reads the specification, weight file, failures, and test reports, then proposes a patch.
- **Evaluation harness:** the repeatable pipeline that decides whether the proposed change helped or broke existing behavior.

The external programmer is not part of ZeroLM's runtime and is not bundled in this repository.

## A practical weight-editing workflow

### 1. Inspect the current design

```sh
python3 scripts/inspect_weights.py
python3 scripts/weight_audit.py
make pipeline
```

Give the weight programmer these files:

- `model/spec.json`
- `weights.zlm`
- `reports/latest.json`
- `tests/regression.tsv`
- `tests/unknown.tsv`

Ask it to explain one concrete failure. Do not ask it to “make the model smarter” without a measurable target.

### 2. Ask for a small patch

The patch format supports three explicit operations:

- `add_feature`: add a phrase and a manually selected weight to an intent.
- `set_feature_weight`: change the weight of one existing phrase.
- `replace_response`: replace one intent's fixed response path.

Example:

```json
{
  "description": "Recognize one additional informal greeting",
  "operations": [
    {
      "op": "add_feature",
      "intent": "greeting",
      "phrase": "yo there",
      "weight": 2.5
    }
  ]
}
```

Validate a patch before applying it:

```sh
python3 scripts/apply_patch.py experiments/patches/example.json --dry-run
```

Applying a patch writes a `.bak` copy of `model/spec.json` first. Then rebuild and evaluate:

```sh
python3 scripts/apply_patch.py experiments/patches/example.json
make pipeline
```

### 3. Keep or reject the change

Save the baseline report before making a change. Compare it with the candidate report:

```sh
cp reports/latest.json reports/baseline.json
# make and test one change
make pipeline
python3 scripts/compare_reports.py reports/baseline.json reports/latest.json
```

The comparison fails when any measured accuracy metric decreases. A passing result is still limited by the size and quality of the test data. Add independent test prompts before trusting a change.

## How to extend further from zero

If the no-training rule is strict, the best next targets are bounded tasks that can be described as small programs:

1. **Better input features:** add word-boundary features for negation, question type, time expressions, numbers, and common command forms. Keep each feature explicit and testable.
2. **Compositional parsing:** create typed slots such as `action`, `object`, `date`, and `time`. Validate extracted values with ordinary code instead of asking the network to guess.
3. **Small algorithmic modules:** hand-design dedicated networks for tasks with a clear input/output contract, such as a small arithmetic circuit, grid movement, or finite-state protocol controller. Test each module independently.
4. **Memory outside the network:** store user facts as structured records and retrieve only relevant records. This gives the agent history without placing the whole history into the neural context. *Implemented as a first step in [ZeroLM Live](LIVE.md): facts are stored outside the network in a separate life file, with decay, a capacity limit and phrase-based retrieval. It does not yet handle paraphrased lookups or typed slots.*
5. **Weight-space patches:** let an external model propose a small weight edit for a known failure. Use deterministic tests and a rollback file to decide whether to keep it.
6. **Independent evaluation:** grow a hidden set of paraphrases, ambiguous requests, and unknown prompts. Never use the same examples for both patch design and final claims.

## What will not scale by hand

Writing a large English vocabulary, semantic representation, grammar, factual knowledge base, and broad reasoning ability into a conventional language model by manually entering numbers is not a practical route. A bigger collection of fixed phrases gives better coverage of those phrases; it does not automatically create a general language model.

The research direction worth testing is whether an external reasoner can program useful small circuits in weight space. For ZeroLM, start with clear subproblems and measurable outcomes. Do not call evaluation “training,” and do not claim general intelligence from a test-set score.

## Reference

Takehi, Rikiya, Jayden Teoh, Alex Zhang, and Omar Khattab. “Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.” November 2026. [Article](https://rikiyat.github.io/blog/weight-space-programming/).

## ZeroLM Next: a controlled patch workflow

The research track (`research/`) has a stricter version of this workflow in `research/tools/wsp.py`: `diagnose` explains a failing parse from the forward-pass trace (which boundary unit fired, which unit wrote which field into which row), `validate` checks a patch's format, names and tensor shapes, `apply` backs up the current weights, applies the patch, evaluates before/after on target and regression suites, records the observed numbers in the patch file and reverts automatically on a regression, `revert` restores the last backup. Patches are human-written JSON (`experiments/next-patches/`); no optimizer is involved. The four patches made so far are described in `research/RESULTS.md`.
