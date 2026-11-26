# Experiment GDM: compiled gated-delta memory against random initialization

Status: pre-registered 2026-11-26 (before any GDM result existed); registration values are in `model/specs/gdm.json`. Audit that led here: `research/decisions/gdm-audit-2026-11-26.md`. Results and verdict: see the "Results" sections at the end, written only after the runs.

Abbreviations at first use: WSP (weight-space programming: writing an algorithm into weights), BPB (bits per byte), BPTT (backpropagation through time), GRU (gated recurrent unit), GDM (gated-delta memory), CI (confidence interval), FLOPs (floating-point operations), RSS (resident set size, the memory a process holds). Formulas and glossary: `docs/math.md`.

## 1. Question

Central hypothesis H0 of the project (`docs/research-hypotheses.md`), narrowed to one testable form:

> **H-GDM.** A bounded fast-weight memory whose tables are *compiled* (initialised by weight-space programming) and then updated by lifetime online learning has a measurable advantage in learning efficiency, retention, generalisation or computational efficiency over *the same architecture with randomly initialised tables*, under equal data, parameters, memory capacity, compute and hyperparameter-search budget.

The comparison that counts is compiled against random initialisation of an identical topology. Beating a plain GRU alone would show that an architecture helps, which is already known for delta-rule memories, not that WSP does.

## 2. Prior work and what is, and is not, new

Verified by search on 2026-11-26 (arXiv records seen):

- **Delta-rule fast weights.** Schlag, Irie and Schmidhuber (arXiv:2102.11174) show that linear attention is an additive outer-product fast-weight memory and that the delta rule raises its capacity. DeltaNet (Yang et al., arXiv:2406.06484, NeurIPS 2024) trains it in parallel over sequence length. **Gated DeltaNet** (Yang, Kautz, Hatamizadeh, arXiv:2412.06464, ICLR 2025) adds a decay gate: $S_t=\alpha_t S_{t-1}(I-\beta_t k_tk_t^\top)+\beta_t v_tk_t^\top$. Test-time training layers (Sun et al., arXiv:2407.04620) and dynamic evaluation (Krause et al., arXiv:1709.07432) also adapt at test time. **The memory rule here is Gated DeltaNet's, single head, unparallelised. It is not novel and is not claimed to be.**
- **Induction circuits and match models.** Induction heads (Olsson et al., arXiv:2209.11895) and PAQ-style match models complete repeated n-grams. A key-value memory written with the previous context as key and the current byte as value *is* an induction mechanism. Not novel.
- **Compiled and structured initialisation.** Tracr (Lindner et al., arXiv:2301.05062) compiles programs into transformer weights for interpretability, with no learning afterwards. Mimetic initialisation (Trockman and Kolter, arXiv:2305.09828) sets attention products in closed form. A 2025 line of work studies how data and initialisation shape induction-head emergence (arXiv:2511.01033). A bounded search found no experiment that compiles an algorithm into a language model's *trainable* fast-weight memory and measures it against random tables of the same shape on bytes from scratch. That is a bounded search, not a proof of absence.
- **Continual online learning.** Online neural compression (NNCP), caches and dynamic evaluation are established; forgetting and retention measures for fast-weight memories under domain shift were not reviewed in depth, so none of the continual-learning or retention readings below claims novelty.

**Genuinely open here (the only candidate contribution):** whether a compiled initial value of an otherwise standard gated-delta memory beats its random-initialised twin, in data efficiency, retention or forgetting, in a byte-level language model learning online from nothing. If the answer is no, the project's remaining contribution is engineering (auditability, snapshots, circuits), which the constitution already says is not novelty.

## 3. Architecture (kept minimal; reuses `bptt.rs`)

The online GRU of `bptt.rs` (embedding $x_t\in\mathbb{R}^{16}$, hidden $h_t\in\mathbb{R}^{d}$, readout over $[h_t, x_t]$, single-pass truncated BPTT over windows of 32 bytes, Adam, gradient clip 1.0, every update logged) gains one optional path. For input byte $c_t$ with history $c_{t-1},c_{t-2},\dots$ and rank $r=32$, lags $m=3$:

- Code table $U\in\mathbb{R}^{256\times r}$, value table $V\in\mathbb{R}^{256\times r}$, decoder $D\in\mathbb{R}^{256\times r}$ (all trainable). Fixed sign tags $s_i\in\{\pm1\}^r$, $i<m$ (constants of the construction, not parameters).
- Query $q_t=\mathrm{norm}\big(\sum_{i<m}s_i\odot U[c_{t-i}]\big)$, key $k_t=\mathrm{norm}\big(\sum_{i<m}s_i\odot U[c_{t-1-i}]\big)$, value $v_t=V[c_t]$, with $\mathrm{norm}(z)=z/\sqrt{\lVert z\rVert^2+10^{-6}}$.
- Gates $\alpha_t=\sigma(a_0+w_a\cdot x_t)$, $\beta_t=\sigma(b_0+w_b\cdot x_t)$.
- State $S_t\in\mathbb{R}^{r\times r}$ (fast weights, persistent, 1,024 floats): $S_t=\alpha_t\,S_{t-1}\,(I-\beta_t k_tk_t^\top)+\beta_t\,v_tk_t^\top$, read $o_t=S_tq_t$.
- Logits: $\ell_t=\ell^{\text{GRU}}_t+\gamma\,D\,o_t$ with learned scalar $\gamma$.

Gradients through $S$ inside a window are exact (the state entering a window is a constant, as $h$ is); they are checked against finite differences (`crates/zerolm-eval/tests/gdm.rs`). Capacity is about $r$ associations; the memory is a short-range cache, by construction.

**Compiled (WSP) initialisation.** $U=V=D$ = one seeded table of unit rows (tag `gdm-code-table-v1`, seed 0). It encodes an algorithm (hash the last three bytes into a position-tagged code, store the next byte's code under it, decode a retrieved code to a byte), no vocabulary, grammar or corpus statistics (constitution invariant 11). Step-zero test: a compiled model's second copy of a repeated random block costs less than 0.7 times the first, a random-table model's does not.
**Random initialisation.** $U,V,D$ independent seeded unit-row tables: same shapes, same row norms, same scalars, no alignment. Everything else (topology, update rule, optimiser, execution code path, parameter count, state size) is identical.

## 4. Arms

| arm | topology | memory | tables | updatable parameters |
|---|---|---|---|---|
| `GRU` | GRU with hidden size chosen so the total parameter count is at least that of the GDM arms | none | seeded | all |
| `GDM-random` | GRU + GDM | on | random | all |
| `GDM-wsp` | same | on | compiled | all |
| `GDM-wsp-off` | same, tables compiled | output not added to logits, tables never change | compiled | GRU part only |

`GDM-wsp-off` isolates what the memory itself contributes (C2); it has the same total parameters and dead memory tensors, and fewer *updatable* ones, by definition. Persistent state is reported separately: the GRU arm's persistent state is $d$ floats, the GDM arms' is $d+1{,}024$. The GRU baseline cannot be matched on parameters and on persistent state at the same time; matching parameters is the registered choice, and the memory-off arm answers whether the extra state, not the init, explains a gain.
Strong statistical baselines are retained: Kneser-Ney 5 alone and the online mixture of each arm with it (as arm G of N1) are reported for every arm.

## 5. Equal budgets

Same seeds, per-seed exposure window and order for all arms; each arm's hyperparameters come from a grid of **4** configurations (GRU and memory-off: learning rate; GDM arms: write-gate bias $b_0$ and read gain $\gamma_0$ in a $2\times2$ grid), 150,000 bytes, one selection seed (9300), scored on the probe role (never the evaluation roles). Mixer rate from a 5-value grid per arm. Selection happens once, in screening; the confirmatory phase reads the frozen choices from `model/specs/gdm.json` and never re-selects. Compute per byte is the same code path; wall time and analytic FLOPs are reported, not equalised.

## 6. Experiments, metrics, evaluation

Data: the manifest's human-written Project Gutenberg fiction and non-fiction and Simple English Wikipedia (`research/corpus/manifest.json`, hashes in the reports). Seed $s$ uses its own contiguous window of the fiction exposure stream (offset a function of $s$), so seed spread includes data variance. Screening evaluates on the **dev** role, the confirmatory run on the **hidden** role (4 windows of 25,000 bytes spread over each role stream). Evaluation never updates weights (the harness asserts the weight hash is unchanged); fast-weight state and $h$ start from zero for each held-out stream and change only by reading earlier bytes of that stream.

- **E1 learning efficiency.** Held-out BPB on fiction and non-fiction at every 100,000 exposure bytes; **P1**: bytes `GDM-wsp` needs to reach `GDM-random`'s final BPB, as a fraction of `GDM-random`'s exposure (1.0 if never reached); **P2**: final mean BPB gain.
- **E2 continual learning.** One snapshot at half the exposure (fiction), then the second half either continues fiction (E1) or switches to Wikipedia (branch restored from the snapshot, which also exercises restoration). **P4**: forgetting $=\mathrm{BPB}_{\text{fiction}}(\text{after Wikipedia})-\mathrm{BPB}_{\text{fiction}}(\text{at the branch})$; the gain is the reduction of forgetting.
- **E3 retention.** Recall probe (`crates/zerolm-eval/src/gdm.rs`): 8 key-value lines (`key value.` with 3-letter random keys and values from a fixed pool of 16 pseudo-words), then $L\in\{0,100,300\}$ bytes of unrelated real text, then 8 queries `key ` scored on the first letter of the value, against 8 control queries with unseen keys. Recall $R=1-\text{bits}_{\text{shown}}/\text{bits}_{\text{control}}$. Conditions: `full`, `reset_h` ($h$ and history zeroed, $S$ kept), `reset_s`, `restart` (both zeroed: weights only), `weights` (online learning during the lines in a throw-away copy, then state zeroed). **P3**: $R$ in `reset_h` at $L=100$ (what the fast-weight memory alone keeps through 100 bytes of unrelated input and a recurrent-state reset). 40 trials per cell.
- **E4 generalisation.** BPB on *novel-context positions* (preceding 10 bytes absent from exposure); and the probe itself, whose pairings are unseen combinations of familiar pool words with fresh keys.
- **E5 resources.** Parameters (total, updatable), persistent state floats, snapshot bytes, train and eval wall time and bytes per second, analytic FLOPs per byte (forward and 3x for training), process peak RSS (a separate single-arm run per arm), update count, mean gradient and update norms, checkpoint blocks.

Statistics: per metric, seed-paired differences; two-sided Student-$t$ interval at $1-0.05/4=98.75\%$ (a family of the four primaries); mean, standard deviation, Cohen's $d_z$, all seed values, sign counts. Intervals are over seeds; they do not cover a different corpus.

## 7. Registered acceptance rule (thresholds provisional until the screening amendment)

Primary rules, all for contrast C1 = `GDM-wsp` over `GDM-random`; a rule is *met* when the mean reaches the floor **and** the interval excludes the no-effect value:

| id | metric | floor | status of the floor |
|---|---|---|---|
| P1 | bytes ratio | $\le 0.80$ (20% less data) | user-supplied provisional floor; N2 pilot used the same |
| P2 | BPB gain | $\ge 0.02$ | user-supplied provisional floor; N2 pilot sd 0.012-0.02 |
| P3 | recall gain at $L=100$, `reset_h` | $\ge 0.15$ | proposed here: 15 percentage points of the control's surprisal |
| P4 | forgetting reduction | $\ge 0.02$ BPB | proposed here, same size as P2 |

A rule *passes* if met **and** `GDM-wsp` also beats `GDM-wsp-off` on that metric (so the advantage is attributable to the memory, not to a side effect of init on the GRU part). **Protected conditions** (all required): `GDM-wsp` no more than 0.01 BPB worse than `GDM-random` on either role; training throughput at least 0.5 times the GRU arm's; every snapshot restoration bit-exact (state, weights and the next 2,000 bytes of execution) and every branch restoration hash-identical; the weight hash unchanged by every evaluation and probe. Prevention of leakage and of hidden advantages: items in section 9.

Verdict, computed by the harness from these rules (confirmatory phase only):

- **PASS**: at least one primary passes and the protected conditions hold. Proceed toward larger validation.
- **PIVOT**: no primary passes, but a primary is *met* for C1 or for C3 (`GDM-wsp` over GRU) with the interval excluding zero: the mechanism helps, the compiled initialisation is not what helps. One alternative is then named, with its reason.
- **FAIL**: otherwise. Stop investing in this mechanism; reconsider the WSP hypothesis.

No threshold or rule is changed after the confirmatory run reads the hidden role. Screening results are exploratory and may change only the floors in section 7 (by a dated amendment appended below) and the grids' *contents* never the budgets.

## 8. Genome and Snapshot (reproducibility contract, `bptt_snapshot.rs`)

- **Genome**: canonical JSON of the architecture (core, dimensions, optimiser, rule, memory configuration), a tensor table (name, offset, length, class, updatable), the state tensors (transient $h$; persistent fast-weight $S$; optimiser moments), initialisation provenance (seed, source tags, SHA-256 of the initial parameter vector) and its own SHA-256. It changes if anything it records changes. No dynamic topology exists.
- **Snapshot**: parameters, Adam moments and step count, $h$, $S$, byte history, prequential counters, closed by a SHA-256. Restoring into a model of the same genome and executing the next bytes reproduces every bit; corrupt or foreign-genome snapshots are rejected (tests in `tests/gdm.rs`).
- **Audit history**: every update is logged (stream offset, bits before, gradient norm, update norm) and checkpoint blocks of 1,000 updates carry the weight hash, targets hash and the list of changed tensors (now including the memory tensors when the memory is on).
- Classes: *model parameters* (learned and fixed), *transient state* ($h$), *persistent neural memory* ($S$), *optimiser state*.

## 9. Validity checklist

Evaluation leakage: separate roles; contamination check; weight hash asserted unchanged around every evaluation and probe; probe queries use a copy of the state per query. Hidden compute or memory advantage: identical topology and code path across GDM arms; persistent state and updatable parameters reported separately; the memory-off arm; selection budgets equal. Cherry-picking: seeds fixed in the spec before running (screening 9301-9303; confirmatory 9401-9408; selection 9300; N2 pilot seeds 9201-9203 not reused), all per-seed values reported. Reproducibility: seeds, window offsets, config and spec hash in the report; Snapshot restoration test per run.

## 10. Known limitations (before results)

Single head, rank 32 (capacity about 32 associations), byte level, at most 10^6 exposure bytes, a GRU of about 20,000 cell parameters, host-side research code (not on the neural ISA, so no mainline claim: invariants 10-13), three-lag context only, probe language is synthetic (evaluation only, never exposure), seeds vary initialisation and data window but not the held-out corpus.

## Screening amendments (before the confirmatory run)
Round 1 (dev, 3 seeds, 300,000 bytes) showed an unequal search (memory arms tuned without lr) and an over-capacity probe; round 2 fixed both (6 configurations per arm including lr; 4 facts with 3-letter values; distractor lengths 0, 10, 30, 100; P3 at L=10 chosen from the untrained positive control). No acceptance threshold changed. Details: `model/specs/gdm.json` (`screening_rounds`), evidence in `research/decisions/gdm-evidence/`.

## Results and verdict (confirmatory, 2026-11-26)
**FAIL.** Compiled over random: held-out BPB gain -0.020 (interval -0.100 to 0.061; 3 of 8 seeds positive), retention gain +0.005, forgetting reduction +0.060 (interval includes 0), bytes ratio 0.55 (interval includes 1; curves non-monotone). The plain parameter-matched GRU was better than both memory arms by 0.117 BPB on all 8 seeds. Full table, limits and the single next step: `research/decisions/novelty-gdm-2026-11-26.md`. Reproduce: the `command_confirmatory` in `model/specs/gdm.json` (about 25 minutes on 32 cores); screening: `command_screening`. Tests: `cargo test --release -p zerolm-eval --test gdm` and `--lib`.
