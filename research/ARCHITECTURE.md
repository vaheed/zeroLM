# ZeroLM Next: architecture

ZeroLM Next is a small recurrent network whose weights are written by hand from closed-form constructions. Nothing is trained, searched or fitted. This document gives the computation exactly; `research/build_next.py` is the executable version of every formula below and `rust/src/next/` is the runtime.

## 1. Pipeline

```
 text ──► tokenizer ──► E[token] ─────────────┐
 (generic)  (generic)   embedding lookup      │ x_t ∈ R^35  (one row of the hand-written lexicon)
                                              ▼
        ┌───────────────────── recurrent cell, one step per token ─────────────────────┐
        │ state: tape T ∈ R^{24×51}, pointer p ∈ R^24, max-time m, error mass e         │
        │                                                                               │
        │ r  = pᵀT                         unbind the current row (role-filler read)    │
        │ hA = relu(A_w[x;r] + A_b)        boundary detector (11 units)                 │
        │ (adv,rev) = A_o hA                                                            │
        │ r' = r ⊙ (1−adv)                 a new row starts empty                       │
        │ p' = p + adv (Sp − p)            pointer moves down one row                   │
        │ hB = relu(B_w[x;r';r;adv;rev;m] + B_b)   role-gated writes + error flags      │
        │ u  = B_o hB                      what to write                                │
        │ T  ← T + p' ⊗ u                  bind: outer product with the pointer         │
        │ m  ← m + relu(Gm·[u;m])          running maximum of row time                  │
        │ e  ← e + B_e·hB + relu(1 − Σp')  error mass, pointer overflow                 │
        └───────────────────────────────────────────────────────────────────────────────┘
                                              │ final tape
                                              ▼
   per row: ex = clamp(ex_w·T_i), hV = relu(V_w[T_i;ex;tf] + V_b), err += V_o·hV       validity layer
   rank_i = Σ_j ex_j·H(4(t_i − t_j + ε_ij)),  perm[k,i] = ex_i·tri(rank_i − k)          ordering layer
   slot_k = Σ_i perm[k,i]·T_i ;  class_f = argmax(dec.f.w·slot_k + dec.f.b)             linear readouts
                                              │
                              accepted iff e + Σerr < 0.5
                                              ▼
 structured operations ──► executor (rust/src/next/world.rs)  ──► new world state
 (JSON, no text)            sees operations only, never the input string
```

"generic" parts are the permitted deterministic runtime operations: tokenization (lowercase, split on whitespace and `, . ;`, drop apostrophes, every other symbol becomes its own unknown token), embedding lookup, tensor arithmetic, argmax and JSON serialization. They contain no word, role or grammar knowledge. `cargo test` scans these files and fails if any lexicon word appears as a string literal.

## 2. What is encoded where

| Knowledge | Location | Form |
|---|---|---|
| Which words exist, and what primitive each denotes (`left`→DIR.left, `twice`→CNT.2, `dye`→ACT.paint, …) | `model/next/spec.json` → tensor `E` (106 rows) | one 35-dim 0/1 vector **per token**. 86 words are lexical, 19 are neutral (zero vector; `that` was removed by patch 003), plus `<unk>` |
| Which role a primitive plays in a clause, in what order, and what is illegal | tensors `A_w, B_w, B_o, V_w` | 11 + 68 + 26 hand-derived ReLU units (section 4) |
| How a boundary advances, reverses and times a row | `A_o`, `S`, `T0`, `Gm_w`, units `time_rev`, `time_fwd` | pointer-shift matrix, big-M gates |
| How to order and read the finished tape | `ex_w, rank_eps, rank_slope, dec.*` | threshold comparator, triangle function, linear readouts |
| What an operation looks like | `decode` entries in `spec.json` → `dec.i.w/b` | one linear readout per output key |
| Grid-world semantics (movement, painting, removal, existence conditions) | `rust/src/next/world.rs` | ordinary code; it receives operations only |

There is **no** mapping from a phrase or sentence to an intent or response. The model contains no response strings at all; output keys and class labels are single tokens such as `move`, `left`, `none`.

Token feature space (35 dims): `ACT`{move,paint,remove}, `TYP`{block,circle,triangle,star}, `COL`{6 colors}, `DIR`{4}, `CNT`{1..9}, and the singletons `NEG, IF, UNLESS, SEQ, AFTER, BEFORE, CLOSE, UNK, STOP` (STOP added by patch 001).

Row (tape) fields (51 dims): `ACT, TYP, COL, NCOL` (paint target), `DIR, CNT, NEG`, the condition slots `CTYP, CCOL, CFLAG, CNEG, CCL`, bookkeeping `PB` (pending "before"), `TIME`, and markers `OPEN, GRP, PEND` added by patches 001, 002, 004. Marker fields never make a row "exist".

## 3. Why this is compositional

* **Role-filler binding.** Writing is the outer product `T += p' ⊗ u`; reading is `r = pᵀT`. With a one-hot pointer both are exact. Because every attribute is written to the row the pointer selects, "red" in the second clause cannot attach to the first object. The `binding` ablation (uniform pointer) destroys this.
* **Same word, different role, decided by network state.** A colour token goes to `COL` (object attribute) if the row has no noun yet, to `NCOL` (paint target) if it has one: units `w_COL.c = relu(x_c − cm − hTYP)` and `w_NCOL.c = relu(x_c + hTYP − cm − 1)`, where `hTYP = Σ r'_TYP`. A noun or colour inside an `if/unless` clause goes to the condition slots instead (`cm = r'_CFLAG − r'_CCL`). The embedding of `red` is identical in all three uses.
* **Order is represented, not assumed.** Each row carries a time. A forward boundary gives the new row `m+1`; a reversing boundary (infix `after`, prefix `before`) gives `t_prev − 0.5`. Execution order is the rank of the times.
* **Length generalization is structural.** The same cell is applied for every token; capacity is 24 rows, beyond which the pointer-overflow detector rejects.

## 4. The units

All gates are exact on 0/1 inputs, so each unit is a Boolean function of the activations written next to it.

**Layer A, boundary (11 units).** `hasACT = Σ r_ACT`, `PB = r_PB`. For each class `c ∈ {SEQ, BEFORE, ACT, NEG, STOP}`:
`bnd_c = relu(x_c + hasACT − PB − 1)` and `bnd_c_pb = relu(x_c + hasACT + PB − 2)`; plus `bnd_AFTER = relu(x_AFTER + hasACT − 1)`. `adv = Σ bnd` and `rev = bnd_AFTER + Σ bnd_c_pb`. Meaning: a connector, or a new action, arriving while the row already has an action closes the row; an infix `after`, or any boundary in a row carrying a pending prefix `before`, reverses.

**Layer B, writes (68 units).**
* Pass-through writes: `w_ACT.a, w_DIR.d, w_CNT.n, w_NEG` (`relu(x)`).
* Role-gated: `w_TYP.t = relu(x_t − cm)`, `w_CTYP.t = relu(x_t + cm − 1)`, `w_COL`, `w_NCOL`, `w_CCOL` as above; `w_CFLAG = relu(x_IF + x_UNLESS − r'_CFLAG)`, `w_CNEG = relu(x_UNLESS − r'_CFLAG)`; condition closers `w_CCL_{close,seq,act,stop} = relu(x_c + cm − 1)`; `w_PB = relu(x_BEFORE − hasACT(r))`.
* Time (big-M gating, `BIG = 1000`, `OFF = 100`): `time_rev = relu(r_TIME + BIG·rev − 0.5 − BIG)`, `time_fwd = relu(m + BIG·adv − BIG·rev + 1 − BIG)`; exactly one of them is non-zero at a boundary and the other is 0. `T0[0,TIME] = OFF` and `m0 = OFF` keep all times positive so ReLU gating is exact.
* Error detectors (each adds 1 to the error mass when its pattern occurs): unknown token, direction without object, direction or count inside a condition, count without direction, action after an object or colour, second condition, chained `after/before` (patch 002).
* Patches 001/004 add `w_OPEN`, `w_PEND_{SEQ,AFTER,BEFORE}`, `w_GRP_{after,before}`.

**Layer V, row validity (26 units, shared by all 24 rows).** `overfull_b = relu(Σ_b T − 1)` for each field group; `move_needs_{object,direction}`, `paint_needs_{object,new_color}`, `remove_needs_object`, `*_forbids_*`, `count_needs_direction`, `condition_needs_object`, `row_without_action = relu(ex − hasACT)`, `dangling_connector`, `dangling_pending`.

**Ordering.** `H(d) = relu(d) − relu(d−1)` with slope 4 and ε_ij = ±0.25 (index tie-break) is an exact step on half-integer time differences; `tri(z) = relu(z+1) − 2relu(z) + relu(z−1)` is an exact one-hot of an integer rank.

**Readout.** Each key is one linear map over the selected row: e.g. for `direction` the logits are `[0.5, T_DIR.left, …, T_DIR.down]` and class 0 (`none`) means "omit". `count` has a gated default: the logit of `1` is `0.5·ACT.move`, so a bare `move … left` yields `count: 1` while `paint`/`remove` yield no count.

## 5. Parameter budget (honest accounting)

`info` reports the dense-equivalent size and how many weights are non-zero:

```
Weights: 24814 slots, 1469 non-zero      (zerolm next info)
```

Every non-zero weight is ±1, a constant of the construction (`BIG`, `OFF`, 0.25, 0.5, slope 4) or an embedding 1. Nothing here is "learned parameters"; the count only describes what the file contains.

## 6. Why this is not (yet) an LLM

It is a deterministic parser for a bounded instruction language, expressed as a recurrent network. It does not model text, predict tokens, represent meaning beyond the listed primitives, or produce free-form language. Its output is a structured operation list drawn from a fixed schema. Its behaviour on the supported language equals that of a conventional recursive-descent parser written for the same grammar (`research/baselines/symbolic.py` agrees with it on the benchmark); what is different is the substrate (matrices), the traceability of every decision, and that the construction can be audited tensor by tensor. See `HYPOTHESES.md` and `LIMITATIONS.md` for what was and was not shown.
