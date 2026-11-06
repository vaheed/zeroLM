# Weight-space programming

This page explains what weight-space programming (WSP) means in ZeroLM, how it relates to the research field that studies neural network weights directly, and what that field implies for our design. It is written for anyone new to the project.

## In one paragraph

Most neural networks get their weights from training: an optimizer adjusts millions of numbers until the network behaves well. ZeroLM is not trained on a corpus in advance, so it needs another way to start with useful abilities. WSP is that way: a **compiler** (`crates/zerolm-wsp`) takes a small, readable description of an algorithm, such as "attend over memory slots, verify the match, decode the stored bytes", and writes the exact weights that make the network compute it. The result is ordinary tensors plus a program over a fixed set of tensor operations (the neural ISA, short for instruction set architecture; see [neural-isa.md](neural-isa.md)). Nothing in it is learned, nothing is hidden in host code, and an independent interpreter reproduces every output from the weights alone.

## What WSP is and is not

| WSP in ZeroLM is | WSP in ZeroLM is not |
|---|---|
| A compiler from typed circuit specifications (`model/prog/*.json`) to weights | A learning algorithm |
| A way to install **algorithms**: selection, attention, verification, cleanup, reachability | A way to install **knowledge**: no vocabulary, word lists, facts, or corpus statistics may appear in a circuit |
| Deterministic and pinned: the SHA-256 hash of every compiled circuit is checked in CI | A source of hidden behavior: every constant is a named tensor in its own ablatable block |
| Editable through versioned, reversible patches (`apply`, `diff`, `revert`) | A replacement for learning from data: online learning must still supply what the model knows |

## Where this sits in the research landscape

1. **Weight space understanding**: the geometry and symmetries of weights (invariance, equivariance), used for compression, optimization, and augmentation.
2. **Weight space representation**: learning embeddings of whole models, used to predict a model's behavior from its weights, retrieve models, and edit them.
3. **Weight space generation**: producing new weights with hypernetworks or generative models (diffusion, flow matching), used for conditional generation, merging, initialization, and faster training.

**ZeroLM's WSP is not covered by that taxonomy.** All three parts *learn from* populations of already-trained weights. WSP *writes* weights from a program, with no trained model anywhere. The closest published work is outside that list: RASP and Tracr, which compile programs into Transformer weights (Weiss et al. 2021, arXiv:2106.06981; Lindner et al. 2023, arXiv:2301.05062), and mechanistic studies that find such circuits inside trained models (Olsson et al. 2022, induction heads). That is why the roadmap describes WSP as a compiler.

Several findings from the field still apply directly to us.

### 1. Weights have symmetries, so raw diffs are not canonical

Many different weight settings compute the same function. The survey lists neuron (or filter) permutations, positive scaling, bias or logit translation, orthogonal rotations, sign flips, and attention-head permutations. Accounting for permutations, Entezari et al. (ICLR 2022, arXiv:2110.06296) and *Git Re-Basin* (ICLR 2023, arXiv:2209.04836) show that independently trained networks often become linearly connected.

**Consequence for ZeroLM.** A patch is a list of value changes against one exact parameterization. Today that is safe: every circuit is compiled into one canonical layout, and each patch names the hash of the base it applies to, so it is refused anywhere else. Once local learning starts adapting circuit blocks (planned for the scalable core), two functionally equal models can differ in many entries. Comparing or merging their patches will then need **alignment** (matching neurons up to permutation, as Git Re-Basin does) before a diff means anything. This is now a stated requirement for self-proposed patches (roadmap revision 2, self-proposed WSP patches).

### 2. Models that read weights should respect those symmetries

Networks that take other networks' weights as input work best when they are equivariant to the weight symmetries: *Equivariant Architectures for Learning in Deep Weight Spaces* (DWSNets, ICML 2023, arXiv:2301.12780), *Permutation Equivariant Neural Functionals* (NeurIPS 2023, arXiv:2302.14040), *Universal Neural Functionals* (NeurIPS 2024, arXiv:2402.05232), and *Graph Metanetworks for Processing Diverse Neural Architectures* (ICLR 2024, arXiv:2312.04501), which treats a network as a graph of neurons.

**Consequence for ZeroLM.** The planned internal patch proposer reads the model's own weights and suggests edits. It should be built in this style, operating on circuits as graphs or equivariant sets of neurons, rather than on flattened weight vectors. Our circuits already have named blocks and an explicit program graph, which gives such a proposer a natural structure to read.

### 3. Generated weights need functional guarantees

The survey's open problems include *controllable weight generation*: "current generation pipelines often produce weights without explicit guarantees on their intended functionality". It also asks for "symmetry-aware projections and reversible editing strategies" as defenses in weight space. *Generative Modeling of Weights: Generalization or Memorization?* (2025, arXiv:2506.07998) warns that weight generators can copy their training models rather than generalize.

**Consequence for ZeroLM.** These risks are why every weight change in ZeroLM must be **reversible** (a patch stores old and new bit patterns and refuses a mismatched base), **verified by behavior** (frozen gates and ablations, never "the file changed"), and **independently reproducible** (the reference interpreter). A future proposer is judged by the same gates as hand-written patches.

### 4. Benchmarks are populations of models

The field evaluates on **model zoos**, large collections of trained networks (*Model Zoos: A Dataset of Diverse Populations of Neural Network Models*, NeurIPS 2022, arXiv:2209.14764). It also notes that most results come from small MLPs and CNNs, and that standard benchmarks are missing.

**Consequence for ZeroLM.** The compiled circuits plus their ablated and patched variants form a small, fully labeled zoo: we know exactly what every block does. That is a useful test bed for the patch proposer and for the byte-permutation test, which checks that circuits encode algorithms rather than English.

## How WSP works in ZeroLM today

```text
model/prog/memory-read.json      typed spec (parameters only)
        |
        v   zerolm-wsp compile (deterministic)
program (JSON graph of ISA ops)  +  program weights (named constant tensors in named blocks)
        |                                   |
        |   SHA-256 pinned in model/prog/pinned.json, checked by `zerolm-wsp check`
        v
engine interpreter  ==bit for bit==  reference interpreter (independent code)
```

| Circuit | Replaces | Blocks |
|---|---|---|
| `memory-read` | memory retrieval: attention, coverage verification, value cleanup, accept decision | select, attend, verify, decode, support |
| `relation-graph` | reasoning: decoding relation slots into subject and object matrices | decode, polarity, support |
| `relation-reach` | reasoning: graph search with shortest proofs | one block per inference operator, depth, probe, support |

**Patches.** A patch (`crates/zerolm-wsp/src/patch.rs`) names its circuit, the SHA-256 of its base and of its result, and runs of edited values with both the old and new `f32` bit patterns. Applying it to anything but its base is refused. Applying and then reverting gives back the original hash exactly. A tampered edit leaves the circuit unchanged.

**Ablation.** Zeroing one block must remove exactly the behavior that block carries. For example, zeroing the memory circuit's `verify` block keeps every exact answer but makes the memory answer every negative query. This is checked on every memory gate run.

## References

- Weight Space Learning survey: arXiv:2603.10090;
- RASP: Weiss, Goldberg, Yahav, *Thinking Like Transformers*, arXiv:2106.06981
- Tracr: Lindner et al., *Tracr: Compiled Transformers as a Laboratory for Interpretability*, arXiv:2301.05062
- Induction heads: Olsson et al. 2022, *In-context Learning and Induction Heads*
- Entezari et al., *The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks*, arXiv:2110.06296
- Ainsworth et al., *Git Re-Basin: Merging Models modulo Permutation Symmetries*, arXiv:2209.04836
- Navon et al., *Equivariant Architectures for Learning in Deep Weight Spaces*, arXiv:2301.12780
- Zhou et al., *Permutation Equivariant Neural Functionals*, arXiv:2302.14040
- Zhou et al., *Universal Neural Functionals*, arXiv:2402.05232
- Lim et al., *Graph Metanetworks for Processing Diverse Neural Architectures*, arXiv:2312.04501
- Schürholt et al., *Model Zoos: A Dataset of Diverse Populations of Neural Network Models*, arXiv:2209.14764
- *Generative Modeling of Weights: Generalization or Memorization?*, arXiv:2506.07998
- Ha, Dai, Le, *HyperNetworks*, arXiv:1609.09106
