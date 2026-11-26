# Experimental design: novelty gate (Gate N)

Pre-registration. Frozen values are in `model/specs/novelty.json` (version 1, `status: "thresholds_frozen_from_n2_pilot_confirmatory_not_run"`). Thresholds were set from pilot seeds only (N2: bytes-ratio floor 0.8, novel-context floor 0.02 BPB, 5 seeds; decision record `research/decisions/novelty-n2-2026-11-26.md`) and are never changed after the final run reads the hidden role.

## Arms

| arm | description | learns | status |
|---|---|---|---|
| A | GRU (gated recurrent unit), random init, single-pass truncated BPTT (windows of 32, Adam) | all weights | implemented (`bptt.rs`) |
| B | Kneser-Ney 5 and PPM-C 5 online counts | counts | implemented (`ngram_lean.rs`) |
| C | compiled circuits only (match, no learned representation) with online confidence counters | counters | implemented as N0 reference (host side) |
| D | A plus compiled match circuit, circuit frozen, learned mixer | A's weights, mixer | not implemented |
| E | A plus the *same circuit architecture*, random init, trained by gradient | all | not implemented |
| F | A plus compiled circuit as the *initial value*, then trained; plus weight-resident memory | all | not implemented |
| G | A mixed with B by an online mixture of experts (`hybrid.rs`: 4 contexts from the GRU's max probability, softmax weights, gradient step on $-\ln p$) | A, 8 mixer numbers | implemented (N1) |
| H | G plus fixed-code match model (the PAQ-style hybrid), v2 confidence rule | A, mixer, counters | implemented (N1) |
| X1 | F with compiled values scrambled (same shapes, same norm) | all | not implemented |
| X2 | A, E, G, H scaled to equal FLOPs (floating-point operations) with F | | not implemented |

Fairness: identical chronological exposure streams; no imported weights; hidden role read once; parameter counts within 5% and memory within 10% of F, else reported at both equal-data and equal-compute; 3 or more seeds (default 5, set from pilot variance); the same hyperparameter budget per arm (an equal number of development-role runs per arm, fixed in the spec before the pilot); hand-authored priors are listed in `research/manifest/seed-knowledge.json` and counted in the arm description (a hand-authored prior is not free).

## The architecture question (Gate N, inside scalable-core)

A coherent core, not unrelated mechanisms: a recurrent state $h_t \in \mathbb{R}^{d}$ and one compiled circuit family acting on a window of past bytes, feeding the state.

- Learned core (GRU baseline): $z_t=\sigma(W_z x_t+U_z h_{t-1}+b_z)$, $r_t=\sigma(W_r x_t+U_r h_{t-1}+b_r)$, $\tilde h_t=\tanh(W_h x_t+U_h(r_t\odot h_{t-1})+b_h)$, $h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$. Shapes: $x_t\in\mathbb{R}^{e}$, $W_\cdot\in\mathbb{R}^{d\times e}$, $U_\cdot\in\mathbb{R}^{d\times d}$. Cost per byte $O(d^2+de+256(d+e))$, state $O(d)$. Stable: gates bounded in $(0,1)$; gradient clip 1.0.
- Match circuit (compiled): keys $k_j=\mathrm{hash}(b_{j-m+1..j})\in\mathbb{Z}_{2^{b}}$ stored with successor $b_{j+1}$ (`hash_rows`, `delta_write`); at time $t$, retrieve the most recent row whose key equals $k_t$ (`top-k` over a similarity), verify match length $\ell$, and emit the one-hot of the successor scaled by a learned gain $g(\ell)\in\mathbb{R}$ per length bucket as an input $u_t\in\mathbb{R}^{256}$ to the readout and as a feature $v_t=g(\ell)\,E[b^*]\in\mathbb{R}^{e}$ to the cell: $x_t' = [x_t; v_t]$. Cost $O(1)$ expected per byte, memory $O(n)$ for $n$ stored positions (a ring of $2^{20}$ rows at the 10^7 scale). Failure: hash collisions and high duplicate rates create false matches; N0 shows the gain is small on fiction, so the expected contribution is at most about 0.01-0.15 BPB.
- Update equations. Compiled weights are initial values. F updates them by the same BPTT step as the core; $g$ is initialized to $\log\frac{\text{hits}+1}{\text{misses}+1}$-like constants of the algorithm (not corpus statistics: all zeros plus a prior hit rate of 0.5, as in `MatchModel::confidence`).
- Alternatives (to be compared at equal parameters in the pilot, not adopted): gated delta rule layer $S_t=S_{t-1}\big(\alpha_t(I-\beta_t k_tk_t^\top)\big)+\beta_t v_tk_t^\top$ with $S_t\in\mathbb{R}^{d_v\times d_k}$ ([Gated DeltaNet](https://arxiv.org/abs/2412.06464)); cost $O(d_vd_k)$ per step. Whether it is better than the GRU at 20,000 parameters is open.
- Every content-dependent step must run through ISA operations (invariant 10) and be replayed bit for bit by `isa_ref.rs`. The host-side `MatchModel` is the reference algorithm only. Compiled weights encode an algorithm (hash, compare, retrieve), not vocabulary or corpus statistics (invariant 11), checked by the byte-permutation test.

## Gate N: decision rule (frozen before the final run)

Primary metrics (decided now; pilot sets the floors):

1. **P1**: bytes of exposure for F to reach the BPB that arm E reaches at 10^6 bytes (lower is better), paired over seeds.
2. **P2**: BPB on *novel-context positions* (positions whose preceding 10 bytes never occur in exposure; computed already in Gate A) at equal exposure and at equal compute.

Protected metrics: overall held-out BPB no more than 0.02 worse than arm G; retention κ of at least 0.5 (as Gate A); throughput no worse than 0.5 times arm G; every update logged.

Outcome rules:

- **NOVELTY SUPPORTED**: F beats each of E, G, H on P1 or P2 with a 95% bootstrap interval of the seed-paired difference excluding 0 and the effect at least the pilot-frozen floor $\delta$; X1 does not reproduce the gain; protected metrics hold; the literature review in `docs/novelty-analysis.md` still distinguishes the mechanism.
- **ENGINEERING SUCCESS ONLY**: F beats A and E but not G or H, or X1 reproduces it.
- **HYPOTHESIS REJECTED**: F is not better than E at equal data and compute under valid conditions.
- **INCONCLUSIVE**: validity checks fail, seeds are too few for the interval, or arms are missing.

Effect sizes are reported as paired mean difference and Cohen's $d_z=\bar\Delta/s_\Delta$ over seeds. Block bootstrap over 10,000-byte blocks within a stream; seed-paired intervals across seeds.

## Budgets and scaling rule

10^5 bytes: stability and implementation. 10^6: pilot (freeze $\delta$, variance, seed count). 10^7: comparative validation (one final run on the hidden role). 10^8: Gate B only if an advantage is shown at 10^6-10^7, or a written, testable reason exists why it appears only at scale. 10^9: only if 10^8 supports it. All negative results and failed configurations stay in `research/decisions/`. The hidden role is read once by the final run.

## Experiment N0 (done)

`cargo run --release -p zerolm-eval --bin zerolm-headroom -- --root data --mixer v2` (schema `research/schemas/headroom-report.schema.json`; CI step in the `real-text` job). Results and caveats: `docs/novelty-analysis.md` and `research/decisions/novelty-n0-2026-11-26.md`.

## Experiment N1 (done, pilot): the bars

`cargo run --release -p zerolm-eval --bin zerolm-hybrid -- --root data --exposure 1000000 --heldout 200000` (schema `research/schemas/hybrid-report.schema.json`; a small smoke run is in the `real-text` CI job). Dev role, 3 GRU seeds (9201-9203), 10^6 exposure bytes, 200,000 held-out bytes per role. Evidence: `research/decisions/novelty-n1-evidence/dev-1e6.json`. Mean held-out BPB over seeds:

| arm | fiction | non-fiction |
|---|---|---|
| A online GRU alone | 2.706 | 2.963 |
| B Kneser-Ney 5 | 2.221 | 2.350 |
| Bm B + match model | 2.171 | 2.205 |
| G GRU + B mixture | 2.196 | 2.321 |
| H G + match model | 2.144 | 2.176 |

Paired gains (BPB, mean over seeds): G over A 0.51 / 0.64; G over B 0.025 / 0.029; H over G 0.052 / 0.145; **H over Bm 0.027 / 0.029** (what the GRU adds to a count-plus-match compressor); Bm over B 0.050 / 0.145. **The bar for any ZeroLM-specific claim at 10^6 bytes is H: 2.144 fiction, 2.176 non-fiction.** Reading: the online GRU alone is 0.5-0.6 BPB behind Kneser-Ney 5 at this budget; it adds only about 0.03 BPB to count-plus-match, which is the whole room a learned core has on this metric at 10^6 bytes.

Caveats, all binding: (1) seeds vary only the GRU initialization; the count model and the match model are deterministic, so per-seed intervals and the $d_z$ over three seeds describe GRU-initialization variance, not data variance (data variance needs more held-out roles). (2) Equal exposure only; equal-compute accounting is not reported (GRU training took about 100 s per 10^6 bytes against seconds for the count model). (3) The selected mixer rate is the grid edge (0.1), and the probe BPB differs by 0.0001 from 0.03; the grid should be widened before the confirmatory run. (4) The mixer and match confidences adapt on the held-out stream, prequentially. (5) The GRU is scored from a fresh state with frozen weights; it does not adapt on held-out text (dynamic evaluation would be a further arm). (6) Dev role, one budget; the hidden role is unread. (7) Arm E also needs a stronger conventional neural baseline than a 20,000-parameter GRU before 'conventional' is a fair word.

## Experiment N2 (done, pilot): the trainable compiled circuit

`cargo run --release -p zerolm-eval --bin zerolm-circuit -- --root data --exposure 1000000 --heldout 100000 --seeds 9201,9202,9203` (schema `research/schemas/circuit-report.schema.json`; a small smoke run is in the `real-text` CI job). Dev role, 10^6 exposure bytes, 100,000 held-out bytes per role, 3 GRU seeds. Evidence: `research/decisions/novelty-n2-evidence/dev-1e6.json`. Arms: A plain GRU (47,594 parameters); A+ GRU with parameters matched to the circuit arms (56,218 vs 55,863); D compiled circuit, frozen; F compiled circuit as initial value, then trained; E same architecture with random tables (E also serves as the scrambled-weights control X1: same shapes, no alignment). Init scalars and the circuit learning-rate multiplier were chosen per arm (F, E) from one grid on a probe-role slice. Mean held-out BPB over seeds (alone / + Kneser-Ney 5 / + KN + match):

| arm | fiction | non-fiction |
|---|---|---|
| A | 2.744 / 2.207 / 2.160 | 3.035 / 2.361 / 2.231 |
| A+ | 2.716 / 2.205 / 2.159 | 3.020 / 2.363 / 2.231 |
| D | 2.718 / 2.200 / 2.156 | 2.976 / 2.349 / 2.222 |
| F | 2.706 / 2.199 / 2.155 | 2.971 / 2.350 / 2.223 |
| E | 2.717 / 2.202 / 2.158 | 2.983 / 2.353 / 2.226 |

Paired gains (BPB, mean over seeds, fiction / non-fiction): F over E 0.011 / 0.013 (non-fiction seed gains 0.010, 0.035, -0.006); F over A+ 0.010 / 0.049 (one seed per role is zero or negative); F over D 0.012 / 0.005; F inside the hybrid over G 0.008 / 0.012; inside H 0.005 / 0.008; a random circuit inside the hybrid over G 0.004 / 0.008. Bytes to reach A's final held-out BPB: A+ 830-925 kB, E 780-820 kB, F 640-810 kB, D 650-800 kB.

Reading: the circuit architecture helps (D beats A by 0.03 / 0.06 BPB with no training of the circuit), but most of the gain over a plain GRU is architecture, not the compiled values: F beats the same architecture with random tables (E) by only about 0.01 BPB, with seed-to-seed spread of the same size, and F is not reliably better than the frozen circuit D. Inside the Kneser-Ney plus match hybrid the whole effect shrinks to 0.005-0.012 BPB, and the random circuit gets about half of it. This pilot does not meet the pre-registered bar for NOVELTY SUPPORTED.

Caveats, all binding: (1) three seeds that vary only GRU initialization; intervals describe initialization variance, not data variance. (2) Dev role, one budget; the hidden role is unread. (3) Training wall time was not equalised (A about 110 s, circuit arms 180-195 s per seed at 10^6 bytes). (4) The selected mixer rate 0.3 is the grid edge; widen before the confirmatory run. (5) The circuit is a host-side reference, not yet compiled onto the neural ISA, so no mainline claim follows. (6) The 20,000-parameter-class GRU is a weak neural baseline.
