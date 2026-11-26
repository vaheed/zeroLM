# Experimental design: novelty gate (Gate N)

Pre-registration. Frozen values are in `model/specs/novelty.json` (version 1, `status: "preregistered_thresholds_pending_pilot"`). Thresholds marked `null` there are set from pilot seeds only and are never changed after the final run reads the hidden role.

## Arms

| arm | description | learns | status |
|---|---|---|---|
| A | GRU (gated recurrent unit), random init, single-pass truncated BPTT (windows of 32, Adam) | all weights | implemented (`bptt.rs`) |
| B | Kneser-Ney 5 and PPM-C 5 online counts | counts | implemented (`ngram_lean.rs`) |
| C | compiled circuits only (match, no learned representation) with online confidence counters | counters | implemented as N0 reference (host side) |
| D | A plus compiled match circuit, circuit frozen, learned mixer | A's weights, mixer | not implemented |
| E | A plus the *same circuit architecture*, random init, trained by gradient | all | not implemented |
| F | A plus compiled circuit as the *initial value*, then trained; plus weight-resident memory | all | not implemented |
| G | A mixed with B (log-linear), the conventional hybrid | A, mixer | not implemented |
| H | G plus fixed-code match model (the PAQ-style hybrid) | A, mixer | not implemented |
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

`cargo run --release -p zerolm-eval --bin zerolm-headroom -- --root data --mixer v2` (schema `research/schemas/headroom-report.schema.json`; CI step in the `real-text` job). Results and caveats: `docs/novelty-analysis.md` and `research/decisions/novelty-n0-2026-11-26.md`. Next step N1: implement arms G and H (needs per-byte distributions from the GRU, which `bptt.rs` exposes through `score`), because they set the bar every ZeroLM-specific claim must beat. N2: the trainable compiled circuit (arms D, E, F, X1).
