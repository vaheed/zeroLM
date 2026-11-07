# Questions newcomers ask

Short, plain answers to the questions people usually ask first. Each answer links to the detailed doc.

## Where is the model saved?

There are two parts.

| Part | Where it lives | Size |
|---|---|---|
| Frozen core weights: byte embedding, recurrent core (GRU, a gated recurrent unit), encoders, compiled circuits | Not stored as a file. Rebuilt from a fixed seed at every start and checked against a pinned SHA-256 hash, so it is bit-identical every time. | about 43,500 core parameters, plus the compiled circuit constants |
| Fact memory, `zerolm memory --dir DIR` | `DIR/memory.zlmc` (after the second save also `DIR/memory.prev.zlmc`, the previous version kept for crash recovery) | 24.0 MB |
| Relation memory, `zerolm reason --dir DIR` | `DIR/memory.zlmc` | 25.4 MB |
| In Docker | the `zerolm-data` volume: `/data/memory/` and `/data/reason/` | as above |

`zerolm run` saves nothing: its `teach`/`ask` ledger lasts for one session only ([bootstrap.md](bootstrap.md)). The checkpoint format and its crash-safe save are described in [memory-format.md](memory-format.md).

## Does the model grow when I talk to it?

**No. The model never gets more numbers; learning and memory change the values of numbers that already exist.** Teach it one fact or 1,200 facts, and the fact-memory file stays exactly 23,955,439 bytes.

Think of a notebook with a fixed number of blank pages. Writing a fact fills a page; the notebook does not get thicker. ZeroLM works like that in three places:

1. **Fact memory.** At start there are 2,048 slots and every number in them is 0. Storing `sky color => blue` overwrites the zeros of one slot:
   ```text
   before:  keys[17] = [0, 0, 0, ..., 0]          (128 numbers)
   after:   keys[17] = [0.12, -0.08, ..., 0.31]   (the same 128 numbers, new values)
   ```
   The checkpoint stores every slot, used or blank, so its size never changes.
2. **Online learning.** The next-byte predictor's readout has 256 × 90 = 23,040 numbers. Every byte it sees nudges them by the delta rule (the change is the prediction error times the input). After 40,000 bytes it predicts unseen text at 2.3 bits per byte instead of 8.0. It still has exactly 23,040 numbers ([online-learning.md](online-learning.md)).
3. **Fast weights.** Many short-lived associations are added into **one** shared matrix, on top of each other. That is why they interfere and fade: about 8 associations are recalled perfectly, about 32 at 75%.

## What happens when the memory is full?

- **Fact memory:** after 2,048 facts, a new write is **refused** and nothing changes. `forget` and compaction free slots again.
- **Relation memory:** holds 4,096 statements about up to 2,048 different names.
- **Fast weights:** never refuse, but old associations blur away.

## How do I see how full it is?

- **File size:** `ls -l DIR` or `du -h DIR`. This stays constant, so it tells you little.
- **Fullness:** the command-line program prints it when it exits:
  - `zerolm memory` prints `memory: N active, N superseded, N deleted, N free slots`;
  - `zerolm reason` prints `reason: N statements, N symbols in memory`.
- **Capacity:** `--verbose` also shows the slot and symbol counts against the capacity.

The number to watch is **how full** the memory is, not how big the file is.

## Why a fixed size?

- **Predictable CPU and memory use.** The model cannot slowly eat the machine.
- **Bounded state.** Nothing grows with the length of a conversation. This is a core design rule; a Transformer's key-value cache, by contrast, grows with every token.
- **Simple, safe saves.** One fixed layout, saved atomically and verified by hash.
- **Measurable limits.** Capacity and interference are explicit numbers we report, not hidden growth.

The price is a hard capacity limit and interference between stored items. The roadmap allows growth only in specific, controlled ways:
- a segmenter circuit that gives newly discovered text units their own embedding rows;
- validated weight patches that add capacity;
- the scaling milestone, which compares two or three model sizes.

## Isn't that different from a brain?

Less than you might think. An adult human brain has about 86 billion neurons, and that number stays largely fixed. Adult neurogenesis exists, but it is small, limited to a few regions such as the hippocampus, and still debated in humans. Most learning changes the connections between neurons instead: about 100 trillion synapses whose strengths change, and which are also made and removed over time.

ZeroLM copies only the first part: it changes connection strengths and never adds connections. Adding connections corresponds to the controlled growth on the roadmap.

## How much knowledge fits in about 25 MB?

Today, surprisingly little: about **a small notebook**.

| Part | Capacity | Real information stored |
|---|---|---|
| Fact memory (24 MB file) | 2,048 facts; key up to 8 words, value up to 47 bytes | at most about 100 KB of actual facts |
| Relation memory (25 MB file) | 4,096 statements, 2,048 names | a few thousand short statements such as "rex is a dog" |
| Fast weights | one shared matrix | about 8 associations perfectly, about 32 at 75% |
| Online learner | 23,040 numbers | enough to model a tiny generated language of about 50 words |

**Why so little for 25 MB?** Each fact is stored as large vectors so it can be found reliably:
- a 128-number key;
- eight 128-number word vectors, so paraphrases still find it;
- a 768-number value;
- all as 4-byte floats.

That is about 8 KB of storage for at most 47 bytes of value. The redundancy buys real guarantees:
- exact recall after a restart (1.000);
- paraphrase matching;
- true erasure on `forget`;
- no damage to one fact from another.

It is a filing cabinet, not compressed knowledge.

**How much could the same size hold?** Studies of trained language models measured a knowledge capacity of about **2 bits per parameter** (Allen-Zhu and Li, *Physics of Language Models, Part 3.3: Knowledge Capacity Scaling Laws*, 2024). At 4 bytes per parameter, 25 MB is about 6 million parameters. That gives roughly 12 million bits, or **about 1.5 MB of pure facts**: about fifteen times what we store now. Reaching it means storing knowledge *distributed* across weights, as trained models do. It is then approximate: facts can blur and interfere, and they are hard to delete cleanly.

**What this means for the plan:** the current memory is deliberately the exact, inspectable kind. A real small language model, the 1.0 goal, also needs the distributed kind. The roadmap plans it in three steps:
- learn from real text (the real-text milestone), after first testing whether local learning can build representations from it at all (the representation-feasibility milestone);
- compare model sizes (the scaling milestone);
- keep the exact slot memory alongside, for facts you teach it directly.

## Can I see a picture of the model?

Not yet. A visual map is planned. It will show every weight group with its size and fill level, the relation memory as a graph of entities and statements, and each compiled circuit as its program graph with its blocks. Until then, `--verbose` on `zerolm memory` and `zerolm reason` shows each tensor written and each step of every answer ([getting-started.md](getting-started.md)).

## Is any of this pretrained, or copied from another model?

No. There are no pretrained weights, tokenizers, or model downloads, and an automated audit checks this on every commit ([research-constitution.md](research-constitution.md)). The only hand-written parts are a small, listed set of algorithms and protocol words. They never include facts, vocabulary, or statistics taken from a corpus ([weight-space-programming.md](weight-space-programming.md)).

## Does it understand English?

No, and we do not claim it does. The current test languages are small and generated. The road to real English goes through learning from licensed, human-written text, with gates that can fail ([claims-and-limitations.md](claims-and-limitations.md), [roadmap.md](roadmap.md)).
