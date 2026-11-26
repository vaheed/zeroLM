# Experiment GDM: compiled gated-delta memory, confirmatory verdict

Status: **FAIL** (computed by the rule registered in `model/specs/gdm.json`, spec SHA-256 `9229c666226bfc99bab8769c98fbf3afb764c85262af8c39f093c9a183ab76c4`). Evidence: `gdm-evidence/` (screening rounds 1 and 2, confirmatory JSON and log). Audit before the run: `gdm-audit-2026-11-26.md`. Design and registration: `docs/gdm-experiment.md`.

## What was run
`zerolm-gdm --phase confirmatory`, hidden role, seeds 9401-9408 (fresh), 10^6 exposure bytes per seed (seed-specific window), 4 arms x 8 seeds, configuration frozen from screening (no re-selection), 98.75% Student-t intervals over seed-paired differences. Thresholds were not changed after screening.

## Results (mean over 8 seeds; gain of X over Y, positive = X better)
| contrast | held-out BPB | with Kneser-Ney 5 | novel-context BPB | forgetting | recall L=10 (state kept, h reset) |
|---|---|---|---|---|---|
| C1 compiled over random | -0.020 [-0.100, 0.061], 3/8 seeds positive | -0.0006 | -0.025 | +0.060 [-0.005, 0.124], 7/8 | +0.005 [-0.026, 0.036] |
| C2 compiled over memory-off | -0.020 [-0.086, 0.045] | -0.001 | -0.027 | +0.048 | +0.019 |
| C3 compiled over parameter-matched GRU | **-0.117 [-0.174, -0.061]**, 0/8 | -0.007 | -0.125 | -0.037 | +0.004 |
| C4 random over GRU | -0.098 [-0.154, -0.041], 0/8 | -0.006 | -0.100 | -0.097 | -0.001 |

P1 bytes ratio (compiled needs this fraction of random's exposure to reach random's final BPB): mean 0.549, interval [0.080, 1.017], 6/8 seeds below 1; not met (upper bound above 1). The curves are not monotone in several seeds (BPB rises after about 700,000 bytes), so the target-reaching measure is ill-posed and the 0.55 must not be read as data efficiency.

Primary rules for C1: P1 not met, P2 not met (mean negative), P3 not met (0.005 against 0.15), P4 not met (0.060 mean, interval includes 0, and the lr-equal condition held). Protected: BPB regression failed (compiled worse than random by more than 0.01 on fiction), throughput 0.81 of GRU (ok), all 32 snapshot restorations bit-exact, weights unchanged by every evaluation and probe. Untrained instrument validation held (compiled recall 0.135 at L=0, random about 0), so the probe can see a working memory; training did not strengthen it (trained recall about 0 for every arm).

## Reading
- The compiled initialization gave no advantage over random tables of the same architecture on any registered primary. Its direction on BPB was negative, with seed spread (sd 0.068) larger than any effect.
- The memory itself added nothing measurable (C2 about -0.02, recall about 0), and a plain GRU matched on parameters was better than both memory arms by 0.12 BPB on all 8 seeds.
- Combined with N2 (compiled over random about +0.01, intervals including 0), two different compiled families give no evidence for H0 at 10^6 bytes.

## Limits that bound this FAIL
Rank 32 (capacity about 32 associations; recall horizon of about 10-30 bytes even untrained), unconditional per-byte writes, one head, host-side research code. The learning rate was chosen at 150,000 bytes and the chosen value (3e-3, the edge of the GDM grid) was noisy at 10^6 bytes; absolute BPB (2.8-3.6) is worse than N2's (2.7), and seed spread is large (exposure windows differ by seed). P1 is ill-posed on non-monotone curves. Per-arm peak RSS runs (`--resource-only`, seed 9401, 200,000 exposure bytes, 1 thread, one process per arm, run after the verdict): GRU 652,488 kB; GDM-random, GDM-wsp and GDM-wsp-off 652,700 kB each. The four peaks differ by 0.03%, so they are dominated by the loaded corpus and the harness, not by the arm; the arms' own state is small (persistent state 104 floats for the GRU against 1,098 for the GDM arms; snapshots 0.87 MB each). These runs did not enter the verdict. Single-process throughput in that run: train 6,555 (GRU), 4,857 (GDM-random), 4,923 (GDM-wsp), 7,603 (GDM-wsp-off) bytes/s; eval 13.3k, 15.9k, 18.6k, 18.6k bytes/s. Parameters and snapshots are in the report; analytic FLOPs are in the report. A larger memory or a learned write gate was not tested; the registered verdict is for the tested mechanism only.

## Decision and the one next step
Verdict FAIL: stop investing in the compiled gated-delta memory, and reconsider H0. **Next step (exactly one): the project lead decides whether to retire H0 now, recording ENGINEERING SUCCESS ONLY and closing the novelty track (option (c) of the N2 record; recommended).** No further experiment is proposed until that decision. Nothing here marks any milestone complete.
