# DCNM review and closure

Status: **CLOSED at the pre-registration stage, before any experiment. No novelty claim. H0 stays retired.**

## What was reviewed

A new, separate exploratory track, working name Drift-Corrected Neural Memory (DCNM), was opened 2026-11-26/27 on branch `research/dcnm-pivot` to ask a hypothesis distinct from H0: whether a continually learning network can preserve, correct, and compose previously acquired knowledge across representation drift by transporting a weight-resident associative memory, using only bounded tensor state and no replay of raw observations. This did not reopen H0 and did not change any milestone status. The four documents produced (`docs/research-pivot-dcnm.md`, `docs/dcnm-literature-review.md`, `docs/dcnm-mathematics.md`, `docs/dcnm-experimental-design.md`) are kept as citable background, same disposition as the N2 and GDM records.

## Why it closes before an experiment runs

The literature review (`docs/dcnm-literature-review.md`) found the proposed mechanism — estimate a linear map between old and new key representations from paired before/after observations, transport the stored key-value memory by congruence, hold the memory as closed-form ridge/least-squares sufficient statistics — already exists as a family: DPCR (ICML 2025), LDC (ECCV 2024), SDC, ADC for the transport and sufficient-statistics form; OWM for the excitation-projector mechanism that avoids overwriting unrelated directions. No part of `v = Mk`, `k_new = Tk`, `M_new = M T^{-1}` is new; that was flagged in the proposal itself as elementary.

The mathematics review (`docs/dcnm-mathematics.md`) further shows exact preservation is provably impossible under arbitrary, non-injective drift with bounded memory (two keys that collapse under drift defeat any post-drift reader regardless of memory size), and exhibits a case where assuming no drift has roughly twice the error of simply abstaining. The only genuinely open pieces identified — a per-association reliability certificate for a superposed memory, an error-accumulation law across repeated drift events, detection of non-injective drift, and whether a frozen key encoder already sidesteps the problem — are narrow refinements of a known mechanism family, not a new mechanism. The most honest outcome label if the proposed smallest experiment were run is KNOWN MECHANISM WITH ENGINEERING VALUE, not a novel result.

This is the fourth attempt in a row, after e-prop (Gate A, local learning rule), N2 (compiled match-circuit prior), and GDM (compiled gated-delta memory), to find a mechanism that gives a measurable edge over matched-budget baselines or over already-known methods, and the fourth to fail to clear that bar — three at the experiment stage, this one at the literature/mathematics stage before an experiment was needed to find out. Running the narrow experiment anyway would cost real time for an outcome already determined by the literature review, which is exactly the kind of avoidable, self-flattering experiment the constitution's falsification discipline exists to prevent.

## Also noted, not actioned

- The proposed transport target (the model's own post-drift key) sits close to constitution invariant 6 ("never use the model's own output as a teacher"); this was never tested against real code, since no experiment ran, but any future revival of this mechanism needs a ruling on that question before implementation, not after.
- Several citations in `docs/dcnm-literature-review.md` are marked as unverified against the primary source (recalled, not fetched); they are kept as leads, not as confirmed claims.
- `docs/dcnm-experimental-design.md` stays marked DRAFT / NOT YET FROZEN and is not run.

## Disposition

- No code was written under this track; nothing in `crates/`, `model/specs/*`, or `research/milestones.json` is touched.
- The exploratory "search for a standalone novel mechanism" framing, as distinct from H0, is retired for the same reason H0 was: four independent attempts under this constitution's constraints (bounded memory, no replay, single-pass online, CPU-only, matched budgets) have not found an edge over known methods. This does not claim no such mechanism exists at a different scale or with different constraints; it claims none was found here.
- Effort returns to the roadmap's own open gate: Phase 10 scaling and the Gate B exit criteria (`research/decisions/gate-a-2026-11-26.md`), where "close the gap to Kneser-Ney order 5 at larger scale" is the next licensed, already-pre-registered test.
