# Research hypotheses

All are falsifiable. None is assumed. Status as of 2026-11-26. "Arm" letters are defined in `docs/experimental-design.md`.

**H0 (central).** Algorithmic priors compiled into neural weight-space programs, combined with single-pass online learning and persistent weight-resident memory, give a measurable advantage over a randomly initialized network under equal data and equal compute.

It splits into hypotheses that can be tested separately:

| id | statement | arms compared | primary metric | falsified if | status |
|---|---|---|---|---|---|
| H1 | A compiled match (induction) prior adds BPB (bits per byte) over a strong count model | B vs. B+match | held-out BPB gain, paired | gain CI includes 0 or is below the pre-frozen floor | **Measured on dev (N0): fiction 0.011 BPB at 10^7 bytes, non-fiction 0.148.** Small; not a ZeroLM-specific result (PAQ match models do this). Hidden-role run pending |
| H2 | Compiled initialization lets a learner reach a target loss with fewer bytes than random initialization of the same architecture | D or F vs. E | bytes of exposure to reach target BPB; novel-context BPB at equal exposure | F needs at least as many bytes as E | not run |
| H3 | The advantage of H2 comes from the algorithm, not from parameter count or compute | F vs. X1 (scrambled compiled weights), X2 (equal compute) | same as H2 | X1 matches F | not run |
| H4 | Compiled-then-trainable beats compiled-frozen and fixed-code | F vs. D-frozen vs. H | same as H2 | F no better than H | not run |
| H5 | Weight-resident memory persists across distribution shift better than learned state | F vs. A, E | retention κ after domain shift | κ(F) not above κ(A) | not run |
| H6 | Validated patches add capability | self-patching milestone | before/after metric | no gain | not started |

Where a hypothesis could be true only at scale, the design says why and how it would be tested at small scale first (`docs/experimental-design.md`, budgets).

## Current position

Nothing here supports a novelty claim. H1 is measured and small. The central comparison (H2, H3, H4) has not been run because arms D, E, F need a trainable compiled circuit inside the learned core, which does not exist yet. Outcome today: **INCONCLUSIVE** for H0, for a stated engineering reason, and **not forced either way**.
