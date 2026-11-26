# Research pivot: Drift-Corrected Neural Memory (DCNM)

Status: DRAFT for human review, 2026-11-27. Branch `research/dcnm-pivot`. Documentation only: no code, no experiment, no spec, no change to `research/milestones.json`, `model/specs/*`, or any closed-track document. Companion documents: `docs/dcnm-literature-review.md`, `docs/dcnm-mathematics.md`, `docs/dcnm-experimental-design.md`.

Abbreviations at first use: DCNM (Drift-Corrected Neural Memory), H0 (the retired central hypothesis), N0/N1/N2 (novelty experiments 0-2), GDM (gated-delta memory experiment), e-prop (eligibility propagation, a local learning rule), BPB (bits per byte), BPTT (backpropagation through time), OGD/OWM (orthogonal gradient descent / orthogonal weight modification), RLS (recursive least squares), ISA (instruction set architecture, the fixed tensor-operation list of `docs/neural-isa.md`).

## 1. Scope and separation from the closed track

DCNM is a **separate, exploratory** track. It does not reopen H0, does not rescue the compiled-versus-random-initialization comparison, and does not change the completion state of any milestone: `research/milestones.json` is untouched (at this commit its statuses are: eight milestones `complete` from the constitution through the uniform engine, `real-text` and the representation-feasibility gate `in_progress`, the remaining seven `planned`). No threshold, seed, or frozen spec of the earlier track is altered. The ZeroLM rules of `AGENTS.md` apply unchanged to anything this track might later produce.

## 2. Phase 1: what the prior novelty track established (preserved, not reinterpreted)

| Item | Recorded outcome | Where it is recorded |
|---|---|---|
| H0 (compiled weight-space priors plus single-pass online learning plus persistent memory beat a randomly initialized network at equal data and compute) | **Retired 2026-11-26; outcome ENGINEERING SUCCESS ONLY; the novelty track is closed.** "Retired" means not supported at this scale and mechanism, not disproved for other mechanisms, memory sizes or exposure; no registered threshold was changed. | `docs/research-hypotheses.md` ("Final position (2026-11-26)"); `research/decisions/novelty-gdm-2026-11-26.md`; commit `4f564a8` "novelty: retire H0, record ENGINEERING SUCCESS ONLY, close the novelty track"; `docs/claims-and-limitations.md` |
| e-prop (local eligibility-propagation rule) at Gate A | **Failed at this scale**: probe gain 0.010 BPB against 1.368 for the backpropagation CONTROL (rho 0.007); validity held; outcome FAIL / PIVOT. Led to the constitution amendment allowing single-pass logged online BPTT (invariant 4) and a rerun that passed (probe gain 1.55 BPB). | `research/decisions/gate-a-2026-11-26.md`, `research/decisions/gate-a-evidence/`; commits `b8e988a` (main run) and `0ff4806` (rerun with the mainline rule) |
| N2 (trainable compiled circuit inside the online GRU) | **Inconclusive** for H0 at the pilot (leaning toward ENGINEERING SUCCESS ONLY): compiled over random 0.011 / 0.013 BPB, within seed spread (sd 0.012 over 3 seeds); H2 not supported at the frozen floors; no scale-up. | `research/decisions/novelty-n2-2026-11-26.md`, `research/decisions/novelty-n2-evidence/`; commits `c84ea50` (experiment), `b502412` (decision record) |
| GDM (compiled gated-delta memory, confirmatory, hidden role, 8 new seeds, 10^6 bytes) | **Verdict FAIL** by the pre-registered rule: compiled over random -0.020 BPB (interval -0.100 to 0.061), retention +0.005, forgetting interval includes 0; a parameter-matched GRU was better by 0.117 BPB. | `research/decisions/novelty-gdm-2026-11-26.md`, `research/decisions/gdm-evidence/`, `research/decisions/gdm-audit-2026-11-26.md`, `docs/gdm-experiment.md`; commits `79ed278` (experiment), `ba3f480` (run logs), `6c0f542`/`79c7dc4` (limits, lint) |
| N0/N1 (headroom and conventional bars) | Measured, small: match prior adds 0.011 BPB over Kneser-Ney 5 on fiction at 10^7 bytes; a hybrid with a match model is the bar. | `docs/novelty-analysis.md`, `research/decisions/novelty-n0-2026-11-26.md`; commits `258dc4f`, `6a6be8b` |

What stands from the earlier track is engineering (online GRU learning, the neural ISA, compiled circuits with reference-interpreter parity, weight-resident memory, exact snapshot restore). Nothing in this document modifies or softens any of the above. In particular, the fact that the GDM memory failed to help byte-level language modeling is not evidence about DCNM, and DCNM's possible success would not be evidence for H0.

## 3. The candidate hypothesis (not an assumed truth)

> A continuously learning neural network can preserve, correct, and compose previously acquired knowledge despite changes in its internal representations, by transporting its persistent associative memory into the evolving representation space, using bounded tensor state and without replaying raw observations.

Formalization in `docs/dcnm-mathematics.md`: read $v=Mk$; drift $k'=\phi(k)$; for linear invertible $\phi=T$, $M_{\text{new}}=MT^{-1}$.

## 4. What phases 2 and 3 found

1. **The mechanism as stated is known.** Linear transport of a stored memory by an estimated feature map is the core of DPCR (ICML 2025; ridge map from old to new features on current data, congruence transport of class covariances and prototypes, closed-form ridge reconstruction), of LDC (ECCV 2024; a learned linear projector is reported to beat an MLP one) and, as a translation-only special case, of SDC (CVPR 2020) and ADC (CVPR 2024). The identity $MT^{-1}$ is elementary; the sufficient-statistics form is RLS; the excitation projector is OWM's. Prevention-by-constraint (OGD, OWM, GPM, Adam-NSCL, AlphaEdit) is the dual strategy. DCNM as stated is **not novel**.
2. **Exact preservation under arbitrary nonlinear drift is impossible, even with unbounded memory**, if the memory is read only through the new encoder (collapse of two keys, Prop 5); and drift in directions the current stream never excites cannot be recovered by any estimator (Prop 4: a family of orthogonal drifts indistinguishable on the window leaves only the mean value; "assume no drift" is then twice as bad as abstaining).
3. **What is achievable** is bounded error under stated assumptions: an exact error identity $M\hat T^{-1}r(k)$, a closed form for the linear case, a ridge variance bound, an accumulation identity that is linear in the number of events for near-isometric steps, and a computable reliability gate with a bounded ($O(d^2)$) ledger. All are elementary; each assumption is listed and each unproven item is labelled HYPOTHESIS.
4. **The demand side is the real risk.** Frozen or slowly moving key encoders (momentum encoders; analytic learning over a frozen backbone; the project's own `memory-encode`, which is frozen and has zero drift) avoid the problem. If keys can be frozen without losing what learned keys buy, DCNM solves nothing.

## 5. Constitution and AGENTS.md compliance, and flags for the human

| Rule | Status for any future DCNM experiment |
|---|---|
| No pretrained weights, no offline pretraining (invariants 1, 2) | Synthetic streams generated in-process; encoder trained online from random initialization. |
| Online gradient only as single-pass truncated BPTT, fully logged (invariant 4, `AGENTS.md` rule 2) | If the encoder is trained, it is by single-pass online SGD with every update logged. |
| Every update logs stream, signal, rule, changed tensors, checkpoint hash (invariant 3, rule 4) | A transport is a memory update and must be logged with the same fields. |
| Hidden evaluation never trains (invariant 5, rule 3) | Evaluation facts are held out; the oracle arm that re-encodes stored raw inputs is evaluation-only and must never be reachable by a non-oracle arm (as the symbolic oracle of the reasoning gate). |
| Model output is never a teacher (invariant 6, rule 3) | **Flag.** The transport target is the model's own current key $k'=f_t(x)$, a deterministic function of the *stream input* and the weights, used as a bookkeeping regression target for the memory, not as a label for learning behavior. This is an interpretation, not a settled question; the project lead should rule on it before any mainline use. The research-arm experiment is outside the mainline. |
| Uniform execution on the neural ISA (invariants 10-13) | Not required for a research arm (the GDM precedent); required before any mainline claim. The ISA has no linear solve; the small inverse can be built from matrix products (for example Newton-Schulz). |
| New weight groups must be declared first (rule 6) | The statistics $A,C$, ledger $L$, cross-statistics and anchor copy are new tensors; they would need `weight-groups.json` entries before mainline use. Not touched here. |
| No learning claim from a changed weight file (rule 7) | Every claim requires the before/after behavioral metric (recall retention), with a no-correction arm as the before. |
| Milestones need command, report schema, CI and uploaded evidence (rule 8) | Nothing is declared complete here; no milestone is created. |

## 6. Phase 4 decision

**Decision (a): a defensible but narrow research gap exists; recommend proceeding to the smallest decisive experiment, time-boxed, scope X below. No outcome label is claimed: no experiment has been run, so NOVEL MECHANISM SUPPORTED, KNOWN MECHANISM WITH ENGINEERING VALUE, HYPOTHESIS REJECTED, and INCONCLUSIVE are all still open.**

**Justification, stated skeptically.**
- The mechanism is known (DPCR, LDC, OWM/RLS algebra). I do not recommend the experiment as a search for a new transport mechanism, and the most likely eventual label, if the experiment runs and works, is **KNOWN MECHANISM WITH ENGINEERING VALUE**. A novelty claim is limited to items that the closest method does not provide: a computable per-association reliability certificate for a *superposed* memory with bounded $O(d^2)$ state, the accumulation law for bounded-statistics transport, and the finding (provable) that the identity fallback can be worse than abstaining.
- The gap is real in the sense that no published method found transports a key-value memory matrix across an online-learned key encoder, and the closest method has no guarantee on transfer to old keys and stores $d^2$ per class. The search was bounded (section 15 of the review), so "not found" is not "absent".
- The experiment is cheap (a small synthetic stream at $d\in\{32,64\}$, seconds to minutes per run on a CPU) and falsifiable, and it answers a question that matters to the roadmap: whether a learned, updating key encoder can feed a persistent memory at all (the "reading into memory" milestone), given that today's memory avoids drift by freezing its encoder.
- Stopping is also defensible. If the human judges that an expected engineering-only outcome is not worth the time, option (b) applies: stop; what would be salvaged is the math document (the accumulation identity, Prop 4 and Prop 5) and the literature map.

**Scope X (what the decisive experiment is).** A host-side, research-arm, synthetic study of one question: *at matched encoder plasticity, does bounded-statistics linear transport with a reliability gate preserve and correct a superposed key-value memory across the drift of an online-trained encoder better than (i) no correction, (ii) a frozen or anchored key encoder, and (iii) prevention by orthogonal projection (OWM-style), and does the gate predict which facts fail?* Included: chronological phases (acquisition, drift, retrieval, correction, restart); arms and metrics of `docs/dcnm-experimental-design.md`; injected-drift calibration cases (linear, mildly nonlinear, collapsing). Excluded: composition of memories (open, not decision-bearing), language data, the neural ISA, scale beyond $d=64$, any mainline integration, any claim about H0.

**Stop rules (written before any code; to be frozen with the design):**
- S1: if the frozen or anchored key arm matches transport on both retention and the benefit of key learning, there is no demand: stop, record HYPOTHESIS REJECTED for the demand side.
- S2: if no correction retains within noise at the drift the online encoder actually produces (slow-drift regime), stop: the problem does not occur at this scale.
- S3: if ungated transport equals gated transport and equals the prevention arm, record KNOWN MECHANISM WITH ENGINEERING VALUE and make no novelty claim.
- S4: if the injected-collapse case is not failed by every non-oracle arm, the instrument is invalid: INCONCLUSIVE.

## 7. What the human must approve before any code is written

1. Whether to proceed (a) or stop (b).
2. The interpretation of invariant 6 for transport targets (section 5).
3. Freezing the design in `docs/dcnm-experimental-design.md` (arms, primary and protected metrics, pilot and confirmatory seeds, thresholds set from pilot seeds only, in the manner of GDM and N2) and a decision record location (`research/decisions/`).
4. Implementation language and location (a host-side crate in the style of `zerolm-gdm`, or a throwaway script kept outside the workspace); the repository rules require deterministic seeded randomness and no new dependencies without justification.

## 8. What is not claimed

No novelty, no performance, no learning, and no result of any kind. All "PROVEN" labels in the math document are elementary linear-algebra facts proved by hand here. The closed forms, the Woodbury inverse, the ridge leave-one-out formula, the sufficient-statistics transport, the accumulation identity, the ledger recursion, and the exact values in Prop 4 (average error $2(N-1)$ for the identity rule, $N-1$ for the mean rule, $N$ for zero, over all permutations for $N=5$) were also checked numerically once in a throwaway script outside the repository (random matrices, double precision); that script is not committed and is not evidence about any model. A property-test suite for the identities should be the first step of any implementation. Citation details marked *memory* in the literature review are unverified.
