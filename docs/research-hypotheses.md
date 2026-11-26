# Research hypotheses

All are falsifiable. None is assumed. Status as of 2026-11-26. "Arm" letters are defined in `docs/experimental-design.md`.

**H0 (central).** Algorithmic priors compiled into neural weight-space programs, combined with single-pass online learning and persistent weight-resident memory, give a measurable advantage over a randomly initialized network under equal data and equal compute.

It splits into hypotheses that can be tested separately:

| id | statement | arms compared | primary metric | falsified if | status |
|---|---|---|---|---|---|
| H1 | A compiled match (induction) prior adds BPB (bits per byte) over a strong count model | B vs. B+match | held-out BPB gain, paired | gain CI includes 0 or is below the pre-frozen floor | **Measured on dev (N0): fiction 0.011 BPB at 10^7 bytes, non-fiction 0.148.** Small; not a ZeroLM-specific result (PAQ match models do this). Hidden-role run pending |
| H2 | Compiled initialization lets a learner reach a target loss with fewer bytes than random initialization of the same architecture | D or F vs. E | bytes of exposure to reach target BPB; novel-context BPB at equal exposure | F needs at least as many bytes as E | **Pilot (N2, dev, 10^6, 3 seeds): F over E 0.011 / 0.013 BPB, bytes ratio about 0.91; below the frozen floor (0.8 / 0.02 BPB) and within seed spread. Not supported.** |
| H3 | The advantage of H2 comes from the algorithm, not from parameter count or compute | F vs. X1 (scrambled compiled weights), X2 (equal compute) | same as H2 | X1 matches F | **Pilot: the random-table control E captures about half of the in-hybrid gain; most of the gain over a plain GRU is architecture (D over A 0.03 / 0.06).** |
| H4 | Compiled-then-trainable beats compiled-frozen and fixed-code | F vs. D-frozen vs. H | same as H2 | F no better than H | **Pilot: F over H 0.005 / 0.008 BPB with a random circuit already at 0.004 / 0.008 over G; F over frozen D 0.012 / 0.005.** |
| H5 | Weight-resident memory persists across distribution shift better than learned state | F vs. A, E | retention κ after domain shift | κ(F) not above κ(A) | not run |
| H6 | Validated patches add capability | self-patching milestone | before/after metric | no gain | not started |

Where a hypothesis could be true only at scale, the design says why and how it would be tested at small scale first (`docs/experimental-design.md`, budgets).

**Bars (N1, dev role, 10^6 bytes):** arm H reaches 2.144 BPB on fiction and 2.176 on non-fiction; the GRU adds about 0.03 BPB to count-plus-match. H2-H4 must be measured against G and H, not only against A and B (`docs/experimental-design.md`).

## Current position

Update (N2 pilot, `research/decisions/novelty-n2-2026-11-26.md`): the central comparison was run at 10^6 bytes and does not support H2-H4 at the frozen floors; the circuit is not scaled. 
Nothing here supports a novelty claim. H1 is measured and small. The central comparison (H2, H3, H4) has not been run because arms D, E, F need a trainable compiled circuit inside the learned core, which does not exist yet. Outcome today: **INCONCLUSIVE** for H0, for a stated engineering reason, and **not forced either way**.
