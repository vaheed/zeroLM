# Representation-learning feasibility (Gate A)

Status: the main run has been made and its outcome is **FAIL / PIVOT** (see **Results** at the bottom). The `representation-feasibility` milestone is **in progress**, not complete: its exit gate also needs the project lead's confirmation of the decision (`research/decisions/gate-a-2026-11-26.md` is a proposal), the evidence uploaded by a CI run (the `representation-gate` job, triggered manually), and the earlier `real-text` milestone, which is not complete.

## The question

The project's central hypothesis is that a network starting from random weights can build **useful internal representations** from one chronological pass over real human-written text, using only local learning rules. "Useful" is defined by a test that a lower training loss or a changed weight file cannot fake: the **representation probe**.

## Arms

Every arm uses the selected gated recurrent unit (GRU) core from the same seeded initialization, the same exposure bytes in the same order, and one pass.

| Arm | What changes during exposure |
|---|---|
| A0 frozen | nothing (an untrained model: 8 bits per byte, BPB) |
| A1 random features | the readout only, by the online-learning milestone's delta rule |
| A2 local representation learning | embedding, GRU, and readout, by e-prop (eligibility propagation) with fixed random feedback |
| A3 backpropagation CONTROL | the same tensors by truncated backpropagation through time (BPTT); never part of the mainline |
| A2-perturbed | nothing during exposure; afterwards the initial core plus a seeded random-direction change with the same per-tensor Frobenius norm as A2's change |

## The local rule (A2)

For each GRU weight $W_{ij}$ (unit $i$, input $j$) an eligibility trace keeps only the diagonal, own-unit part of the recurrent dependence:

$$\varepsilon_t^{ij} = (1 - z_{t,i})\,\varepsilon_{t-1}^{ij} + \left.\frac{\partial h_{t,i}}{\partial W_{ij}}\right|_{h_{t-1}\ \text{fixed}},\qquad \ell_t = B\,(\mathbf{1}_{y_t} - p_t),\qquad \Delta W_{ij} = \eta\,\ell_{t,i}\,\varepsilon_t^{ij}.$$

$z_t$ is the update gate, $p_t$ the readout's predicted distribution, $y_t$ the observed next byte (the only learning signal), and $B \in \mathbb{R}^{d\times 256}$ a fixed seeded random matrix (feedback alignment). Nothing flows backward through time or through the transposed readout. For a weight in unit $i$'s own row, the one-step term $\partial h_{t,i}/\partial W_{ij}$ is exact (checked against finite differences for every GRU tensor). For the reset gate only the own-unit path into the candidate is kept, and the embedding row of the input byte uses the one-step signal without a trace, so the rule is an approximation by design and not exact RTRL (real-time recurrent learning). The per-step update norm is bounded and every update can be logged. Code: `crates/zerolm-learning/src/plastic.rs`; weight group `plastic_core` in `model/specs/weight-groups.json`.

Checked by `crates/zerolm-eval/tests/eprop.rs`: traces match a per-weight finite-difference evaluation of the recursion; on a tiny network the summed e-prop direction has cosine 0.98 with the exact (BPTT) gradient when the feedback is the transposed readout, and about 0 with random feedback (feedback alignment has to build that alignment during training); updates are deterministic; a disabled model changes nothing; the bits for a byte do not depend on later bytes; the perturbation has exactly A2's per-tensor norms.

## The decisive test: the representation probe

After exposure, freeze an arm's embedding and core, discard its readout, train a **fresh** readout with the same delta rule and constants in one pass over the probe split $P$ (whole documents not in exposure), then score the hidden split $H$:

$$\mathrm{BPB}^{\text{probe}}_x = \mathrm{BPB}_H\big(\text{fresh readout on } P \mid \text{core of arm } x\big),\qquad \Delta R_x = \mathrm{BPB}^{\text{probe}}_{A1} - \mathrm{BPB}^{\text{probe}}_x,\qquad \rho = \frac{\Delta R_{A2}}{\Delta R_{A3}}.$$

$\Delta R_x > 0$ means arm $x$'s core is a better feature extractor than random features with the readout held fixed. $\rho$ is the fraction of the available representation gain that local learning captures.

## Data

Two exposure sources give the retention test a domain shift: 33 public-domain fiction books ($E_1$), then 23 public-domain non-fiction books ($E_2$) (`research/corpus/manifest.json`, `docs/data.md`). Splits are by whole document. The probe split is about 10^6 bytes, the hidden split 2 x 10^6 bytes, and the exposure stream is $5\cdot10^6 + 5\cdot10^6$ bytes. The contamination check (13-word windows) covers $P$ and $H$ against exposure.

Other measurements: prequential BPB (each byte scored before it is learned) and held-out BPB at $10^5$, $10^6$, $5\cdot10^6$ (end of $E_1$), and $10^7$ bytes; BPB on novel-context positions of $H$ (the preceding $k = 10$ bytes never occur in exposure; about 46% of $H$); retention $\kappa = \Delta R^{(E_1)}_{\text{after }E_2} / \Delta R^{(E_1)}_{\text{after }E_1}$ (the probe gain on first-source text, measured after each segment; undefined when the gain after $E_1$ is below $r$); expected calibration error (ECE); stability and resources.

## Decision rule (frozen in `model/specs/representation.json`)

Significant means mean minus two standard deviations above the bound, across the 3 main seeds.

- **Validity (any failure is INCONCLUSIVE):** V1 pipeline (hashes, contamination, determinism, disabled means no update, no future leak); V2 headroom ($\Delta R_{A3}$ significant and at least $h$); V3 probe validity ($\Delta R$ of A2-perturbed not significantly positive); V4 baselines sane.
- **PASS (all):** P1 $\Delta R_{A2}$ significant, at least $r$, above the perturbed control; P2 $\rho \ge \rho_{\min}$; P3 A2 beats A1 end-to-end on novel-context positions; P4 $\kappa \ge \kappa_{\min}$; P5 no non-finite value and excess late-window degradation at most $s$.
- **FAIL / PIVOT:** validity holds and any of P1-P5 fails; the project lead then records a decision in `research/decisions/`.

The spec's `history` field keeps every pilot version, including the failed ones.

## Running it

```
cargo run --release -p zerolm-eval --bin zerolm-representation -- --root data --mode smoke   # tiny end-to-end check; outcome is always "smoke"
cargo run --release -p zerolm-eval --bin zerolm-representation -- --root data --mode pilot   # pre-registered configurations, development role, pilot seeds
cargo run --release -p zerolm-eval --bin zerolm-representation -- --root data --mode main    # needs the frozen spec; reads the hidden role
```

Reports: `reports/representation/{summary.json,brief.md}` (schema `research/schemas/representation-report.schema.json`).

## Results

**Gate A outcome: FAIL / PIVOT** (main run 2026-11-26, commit `e4a4d90` on a clean tree, spec version 3, 3 fresh seeds 20261201-20261203, hidden split read once). Deciding criteria: P1, P2, P4. All validity checks (V1-V4) held, so this is a decision-grade negative result for the primary candidate at this scale, not an engineering failure. The full report is `research/decisions/gate-a-evidence/main-summary.json` (copied from `reports/representation/`, which is git-ignored); the run took about 35 minutes of wall time on 32 cores.

Final probe results, means over the 3 seeds (bits per byte on the hidden split; lower is better):

| arm | probe BPB | $\Delta R$ (per seed) | end-to-end held-out BPB | novel-context BPB | ECE |
|---|---|---|---|---|---|
| A0 frozen (zero readout) | 4.516 | 0 | 8.000 | 8.000 | 0.004 |
| A1 random features | 4.516 | 0 | 4.823 | 5.071 | 0.25 |
| A2 e-prop | 4.506 | +0.016, +0.009, +0.005 (mean 0.010) | 4.810 | 5.053 | 0.25 |
| A3 backpropagation CONTROL | 3.148 | +1.359, +1.422, +1.322 (mean 1.368) | 2.905 | 3.211 | 0.09 |
| A2-perturbed | 4.522 | -0.002, -0.006, -0.009 | | | |

- **V2 headroom holds:** the control's core is a far better feature extractor than random features ($\Delta R_{A3} = 1.37$ BPB, 9x its seed spread). V3 holds: the matched random perturbation gains nothing.
- **P1 fails:** $\Delta R_{A2} = 0.010 \pm 0.006$ (mean $\pm$ standard deviation), not significant (mean minus two standard deviations is below 0), and ten times below the frozen floor $r = 0.1$.
- **P2 fails:** $\rho = 0.007$: local learning captured under 1% of the representation gain that backpropagation obtains from the same architecture, initialization, and bytes.
- **P3 holds but is tiny:** A2 beats A1 on novel-context positions by 0.019 BPB.
- **P4 fails (undefined):** the probe gain on first-source text after $E_1$ was 0.003 (A2), below the floor under which the ratio $\kappa$ means anything; the backpropagation control retains $\kappa \approx 1.0$ (the gain on fiction survives training on non-fiction).
- **P5 holds:** all values finite; excess late-window degradation 0.016 BPB.
- **Baselines on the same stream (hidden split):** Kneser-Ney order 5 2.18 BPB, PPM-C order 5 2.21, Witten-Bell order 5 2.25. Even the backpropagation CONTROL (held-out 2.91) is far behind the count-based baselines at 20,000 cell parameters and 10^7 bytes.

### Pilot history (development role, pilot seeds 9101 and 9102, 5·10^5 + 5·10^5 exposure bytes)

Reported in full in `research/decisions/gate-a-evidence/pilot-v1.md` and `pilot-v2.md` and in the `history` field of the spec.

- **Version 1** pre-registered learning rates $\eta \in \{0.003, 0.01, 0.03\}$ at two core sizes (20,000 and 200,000 cell parameters). Every configuration was unstable or harmful at 20,000 parameters (probe $\Delta R$ from -0.02 to -1.1); the backpropagation control gained 1.1 BPB (20,000) and 1.3 BPB (200,000). One 200,000-parameter run (large $\eta$, seed 9101) had a *positive* probe gain of 0.21 although its own end-to-end readout diverged (held-out 82 BPB); the other seed lost 2.8 BPB, so no pattern is claimed.
- **Version 2** pre-registered three smaller rates ($3\cdot10^{-5}$ to $3\cdot10^{-4}$): no gain (-0.015 to 0.000). The smallest was frozen.

### How to read this result

What it supports: with this rule, this core, this corpus, and 10^7 bytes, e-prop with fixed random feedback does not build better features than random ones, while backpropagation from the same start does. Under the roadmap's decision table (P1 fails while A3 shows clear headroom), the central hypothesis is **not supported at this scale**; the choices are (a) allow single-pass online gradient updates in the mainline, (b) narrow the objective to a compressor-grade system, or (c) discontinue. The recommendation and the decision are in `research/decisions/gate-a-2026-11-26.md`.

What it does not show: that no local rule can work. Only one rule was tried (the optional second, predictive rule was not pre-registered); the e-prop learning rate was searched over six values on two pilot seeds; the diagonal approximation drops cross-unit terms by design; the core is small (20,000 parameters; 200,000 was piloted only with the version-1 rates); the stream is 10^7 bytes of one language and one genre family. A caveat on the end-to-end numbers: the delta-rule constants for the readout were tuned on a generated language, and A1's own held-out BPB (4.82) is *worse* than byte frequency (about 4.5) and gets worse with exposure; that is why the decisive test uses a fresh readout and the same constants for every arm.

Cost (reference machine, per seed): A1 6,300 bytes/s, A2 4,800 bytes/s (about 126,000 multiply-adds per byte, analytic), A3 5,700 bytes/s. Peak resident set size is process-wide (all arms and seeds in one process): 1.4 GB.

## Rerun with the mainline online-gradient rule (option (a))

After the FAIL / PIVOT above, the project lead chose option (a) on 2026-11-26: constitution invariant 4 now allows single-pass online gradient learning in the mainline (still no offline pretraining; every update logged; see `docs/research-constitution.md`, `research/decisions/gate-a-2026-11-26.md`, and the audit's `online_gradient` check in `research/manifest/update-policy.json`). The gate is rerun with that rule as the mainline arm A2:

- **A2 mainline:** the same seeded core, trained in one chronological pass by truncated backpropagation through time (windows of 32 bytes, Adam, gradient-norm clip 1.0, learning rate 0.001). Each window is scored with the weights from before its update, then one step is taken. Every update is logged (offset, bytes, bits before the update, gradient and update norm) and closed into checkpoint blocks carrying the stream name, the signal `observed_next_byte`, the rule `truncated_bptt_adam`, the changed tensors, a hash of the targets, and the weights' SHA-256. Weight group `gradient_core`.
- **Data:** same exposure stream, probe split, and 20,000-parameter core; the held-out role is two source groups held back whole and never seen by any run (`gutenberg-fresh-fiction`, `gutenberg-fresh-nonfiction`), and the seeds are new (20261211-20261213; the earlier seeds are listed as burned in the spec).
- **Spec:** `model/specs/representation-gradient.json`, frozen before the run. Criteria: V1 (pipeline, including an exact update log), V3 (perturbed control), V4 (baselines); P1 probe gain at least 0.5 BPB, significant and above the perturbed control; P2 end-to-end held-out BPB at least 1.0 below byte frequency; P3 gain over A1 on novel-context positions; P4 retention $\kappa \ge 0.5$; P5 finite, bounded weights, and no regression of held-out BPB above 0.25 between milestones. There is no headroom check any more because the headroom arm is now the mainline arm.
- **Honest status of this run:** the rule is the old CONTROL, whose probe gain on the earlier hidden split (1.37 BPB) and the pilots was seen when the thresholds were set (the spec's rationale says so). The rerun confirms it under mainline rules on unseen text with new seeds and a verified update log; it is not an independent discovery. A PASS permits Phase 10 to plan with online gradient learning; it does not show the model reaches Level L1, since the same arm's held-out BPB (about 2.9) is far above Kneser-Ney order 5 (about 2.2).

```
cargo run --release -p zerolm-eval --bin zerolm-gradient-gate -- --root data --mode smoke   # tiny end-to-end check
cargo run --release -p zerolm-eval --bin zerolm-gradient-gate -- --root data --mode main    # reads the fresh held-out books once
```
