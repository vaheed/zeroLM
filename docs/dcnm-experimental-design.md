# DCNM experimental design (SKELETON)

**DRAFT. NOT YET FROZEN. NOT PRE-REGISTERED.** No numerical thresholds, seed lists, hyperparameter grids or sample sizes are fixed here; they will be set from pilot seeds only, after the human approves proceeding (`docs/research-pivot-dcnm.md`, section 7), and frozen in a spec before any confirmatory run, in the manner of `docs/experimental-design.md` and `docs/gdm-experiment.md`. This file only says what the smallest decisive experiment would look like. Math: `docs/dcnm-mathematics.md`.

Abbreviations at first use: DCNM (Drift-Corrected Neural Memory), RLS (recursive least squares), OWM (orthogonal weight modification), SGD (stochastic gradient descent), AUROC (area under the receiver operating characteristic curve), ISA (instruction set architecture, `docs/neural-isa.md`).

## 1. Question

At matched encoder plasticity, does bounded-statistics linear transport with a reliability gate preserve and correct a superposed key-value memory across the drift of an online-trained encoder better than no correction, a frozen or anchored key encoder, and prevention by projection; and does the gate identify which facts fail?

## 2. Setup (small, synthetic, CPU)

- **Inputs** $x\in\mathbb{R}^{d_x}$ drawn from a seeded generator with a controllable nuisance factor (so that learned keys can become more robust than random ones).
- **Encoder** $f_\theta$: a small multilayer perceptron with normalized output key $k\in\mathbb{R}^{d}$, $d$ in a small range (for example 32 and 64; values to be fixed in the spec), trained online from random initialization by single-pass SGD on an auxiliary task (invariance to nuisance, or a next-element task on the stream). Drift is therefore produced by learning, not injected. Every update logged.
- **Memory**: $M\in\mathbb{R}^{m\times d}$ written by the RLS rule with sufficient statistics $(A,C)$ (`docs/dcnm-mathematics.md`, section 2). Facts: $N$ pairs $(x_i\to v_i)$ with $v_i$ from a random codebook; load $N/d$ is an experimental axis, kept below the exact-capacity limit for the main comparison.
- **Injected-drift mode** (calibration only): known $T$ applied to keys: near-identity linear, orthogonal, anisotropic, mildly nonlinear, and **collapsing** (rank-deficient or non-injective). Used to validate the instrument and to test the gate.

## 3. Chronological phases (one pass, in order)

1. **Acquisition.** Encoder warmed up on the stream; $N$ facts written at time $s$. Pre-drift recall recorded (the "before").
2. **Drift.** The encoder keeps learning for a stretch of the stream that never contains the fact inputs; the fraction of stream energy lying in the facts' key subspace (excitation overlap) is an axis. At each consolidation (interval $w$ steps) every arm applies its correction using only the current stream's before/after key pairs.
3. **Retrieval.** Fact inputs are queried through the current encoder (evaluation-only access to the raw fact inputs). Recall, margin, abstentions.
4. **Correction.** A subset of facts is corrected after drift by an ordinary write through the new key. Measure: corrected value returned, stale value not returned (ghost rate), uncorrected facts undisturbed.
5. **Restart.** Serialize $(M,A,C,L,\theta,\text{anchor})$, reload, verify bit-exact state and identical metrics, then continue drifting briefly. Mirrors the repository's exact snapshot-restore property.
6. *(Secondary, exploratory, not decision-bearing.)* After drift, write new facts and re-check old ones (interference); an optional two-hop composition probe ($a\to b$, $b\to c$) is deferred.

## 4. Arms

| Arm | Description | Role |
|---|---|---|
| N0 | no correction (memory untouched) | the "before" for any learning claim |
| F | frozen key encoder (keys never drift), or equivalently queries read through an anchor snapshot of the key head | avoids drift; shows the cost in key quality |
| P | prevention: encoder updates projected orthogonally to a bounded key-input covariance (OWM-style) | the dual strategy |
| TU | transport, ungated: ridge-to-identity $\hat T$ from current-stream pairs, congruence transport of $(A,C)$ | DPCR-equivalent core |
| TG | transport with gates G1-G3 and ledger-based abstention (G4) | the candidate |
| TS | transport with shuffled pairs (breaks the correspondence) | control that transport uses the information |
| TI | transport from injected oracle $T$ in calibration mode | instrument check (recall should be near the pre-drift level) |
| O | oracle: re-encode stored raw inputs and rebuild the memory | evaluation-only upper bound; never reachable by another arm; excluded from strict results |

Fairness: identical streams and seeds; the same encoder updates for all arms except F and P (which change what is allowed); equal persistent-state budget reported per arm (bytes); the same tuning budget per arm on pilot seeds only.

## 5. Metrics (names only; thresholds to be set from pilot seeds)

Primary:
- **P1 retention**: old-fact recall after drift divided by pre-drift recall, as a function of drift magnitude and excitation overlap.
- **P2 key-learning benefit**: recall of **new** facts under query perturbation, written after drift, compared with arm F. This is the demand-side test: if F matches TG here and on P1, there is nothing to solve (stop rule S1).

Protected:
- encoder auxiliary-task loss (plasticity not worse than N0 by a margin to be set);
- precision (wrong answers on negative queries) and abstention behavior;
- persistent state size constant in $N$ and in the number of drift events (bytes recorded);
- update log completeness and restart bit-exactness;
- wall time per consolidation.

Diagnostics (reported, not decision-bearing): gate calibration (AUROC of the ledger $\hat u(k)$ and of $c_M$ for predicting facts that fail); measured accumulation versus the linear-in-$\tau$ law; relative size of accumulated drift to key separation (regime map for hypothesis H7); fraction of drift energy in the excited subspace ($c_M$).

## 6. Validity checks and controls

- **Positive instrument check** (TI): injected linear $T$ recovers recall to near pre-drift level, so the probe can see a working transport.
- **Negative instrument check**: injected collapse must be failed by every non-oracle arm; if not, the instrument is invalid (INCONCLUSIVE).
- **No-leak audit**: non-oracle arms receive only current-stream key pairs; raw fact inputs reach only the evaluation harness and O; checked by interface.
- **Identity property tests** before any run: Prop 1 (transport equals re-encoded ridge memory), the error identity of section 3.2, and the accumulation identity of section 4 of the math document hold to numerical precision on random matrices.
- Pilot seeds set variance and thresholds; confirmatory seeds are fresh and read once; no re-selection after the confirmatory run; all failed configurations stay on record.

## 7. Axes (values to be fixed in the spec)

drift magnitude (learning rate times steps); excitation overlap between the stream and the fact subspace; load $N/d$; window $n$ relative to $d$; consolidation interval $w$; ridge $\lambda$; gate thresholds; drift type in calibration mode.

## 8. Decision rule (to be written with numbers at freeze; the shape only)

- **Supports DCNM as engineering value**: TG beats N0, F and P on a pre-set combination of P1 and P2 with seed-paired intervals excluding zero and an effect at or above a pilot-set floor, protected metrics hold, TS fails, validity checks pass.
- **Known mechanism with engineering value**: TG is not distinguishable from TU, or TU matches P. No novelty claim.
- **Hypothesis rejected (demand side)**: F matches TG on P1 and P2, or N0 retains within noise at the drift the encoder produces (stop rules S1, S2 of the pivot document).
- **Inconclusive**: instrument or validity failures, too few seeds, missing arms.

## 9. Constraints and implementation notes

Host-side research arm (the `zerolm-gdm` precedent); deterministic seeded randomness; no new dependencies without justification; if the encoder is trained, single-pass online SGD with every update logged (constitution invariants 3 and 4); the transport is a logged memory update. No new weight group, ISA program or milestone is created by this experiment; any mainline use requires `weight-groups.json` declarations, an ISA compilation (the ISA has no linear solve, so a small inverse would be built from matrix products), a ruling on invariant 6 for transport targets, and its own milestone with command, schema, CI job and evidence.

## 9a. Open items before freezing

Which generator for the nuisance task; the exact injected-drift families; seeds; pilot sample sizes; where the report schema and evidence live (`research/decisions/`, `research/schemas/`); the decision-record template name. None is decided here.
