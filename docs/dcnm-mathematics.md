# DCNM mathematics: transport of a bounded associative memory under representation drift

Status: DRAFT, 2026-11-27. No experiment has been run. Part of the exploratory track described in `docs/research-pivot-dcnm.md`; literature positioning is in `docs/dcnm-literature-review.md`. Every statement is tagged **PROVEN** (a proof or a cited theorem is given or sketched in full), **DERIVED** (follows from proven pieces plus an explicit assumption), or **HYPOTHESIS** (to be tested; not established). Nothing here is evidence about ZeroLM behavior.

## 0. Terms and notation

| Term | Meaning |
|---|---|
| DCNM | Drift-Corrected Neural Memory: the candidate named in the research pivot. |
| Key, value | $k \in \mathbb{R}^{d}$ (column vector, produced by the encoder) and $v \in \mathbb{R}^{m}$. |
| Memory | $M \in \mathbb{R}^{m \times d}$; read is $\hat v = M k$. |
| Drift | A change of the encoder from $f_s$ (at write time $s$) to $f_t$ (now): the same input $x$ gets key $k' = \phi(k)$, where $k = f_s(x)$, $k' = f_t(x)$. |
| Transport | Changing $M$ so that it reads correctly from the new keys. |
| RLS | Recursive least squares: the exact online solution of (ridge) linear regression by a rank-one update of an inverse covariance (Sherman-Morrison identity). |
| Ridge | Least squares with an added penalty $\lambda \lVert\cdot\rVert_F^2$. |
| SVD | Singular value decomposition. |
| Loewner order | $A \preceq B$ means $B-A$ is positive semidefinite. |
| LOO | Leave-one-out (cross-validation residual). |
| FD | Frequent Directions, a deterministic streaming matrix sketch (Liberty 2013). |
| OWM, OGD | Orthogonal weight modification; orthogonal gradient descent (see the literature review). |
| ISA | Instruction set architecture: ZeroLM's fixed list of tensor operations (`docs/neural-isa.md`). |
| $\lVert\cdot\rVert$ | Euclidean norm for vectors, operator (spectral) norm for matrices unless subscripted $F$ (Frobenius). |

Window of $n$ paired observations from the **current** stream: $K = [k_1,\dots,k_n] \in \mathbb{R}^{d\times n}$ computed by the pre-update (or anchor) encoder, $K' = [k'_1,\dots,k'_n]$ by the current encoder on the same inputs, $D = K'-K$. No stored fact input is ever used. Keys are assumed norm-bounded, $\lVert k\rVert \le 1$ (the encoder normalizes).

## 1. The elementary identity (NOT the contribution)

If the drift is linear and invertible, $\phi(k) = Tk$ with $T \in \mathbb{R}^{d\times d}$, then
$$M_{\text{new}} = M\,T^{-1}\quad\Rightarrow\quad M_{\text{new}}\,(Tk) = M k .$$
One line of algebra, **PROVEN**, and not a research result. If $T$ is singular, $M_{\text{new}} = M T^{+}$ (Moore-Penrose) gives $M_{\text{new}}Tk = M P_{\mathrm{row}(T)}k$, i.e. exactly the part of $k$ in $\ker T$ is lost (those keys collapsed under drift; see Prop 5).

Everything difficult is in **estimating $T$** from data that excludes the old inputs, and in knowing when the estimate may not be trusted.

## 2. Bounded-statistics form: the same identity as exact ridge regression (PROVEN)

Write the memory as the ridge solution over its $N$ stored pairs $(k_i,v_i)$, $K_0\in\mathbb{R}^{d\times N}$, $V\in\mathbb{R}^{m\times N}$:
$$M_\lambda = A\,(C+\lambda I)^{-1},\qquad A = V K_0^{\top}\in\mathbb{R}^{m\times d},\quad C = K_0K_0^{\top}\in\mathbb{R}^{d\times d}.$$
$(A,C)$ are sufficient statistics: a write adds $vk^\top$ to $A$ and $kk^\top$ to $C$ and costs $O(d^2)$ with a Sherman-Morrison update of $(C+\lambda I)^{-1}$. Their size does not depend on $N$.

**Prop 1 (transport of sufficient statistics).** If the drift is $K_0' = TK_0$ with $T$ invertible, then
$$A' = A T^{\top},\qquad C' = T C T^{\top},\qquad M'_\lambda = A'(C'+\lambda I)^{-1}$$
equals *exactly* the ridge memory that re-encoding every stored input would produce. (Proof: substitute $K_0'=TK_0$ into the definitions.) For $\lambda\to 0$ and $K_0$ square invertible this is $M T^{-1}$. For $\lambda>0$ it is **not** $M_\lambda T^{-1}$ unless $T$ is orthogonal: the penalty must be applied in the new coordinates. For $N<d$ with independent stored keys, both $M T^{-1}$ (minimum-norm in old metric) and $M'_0$ (minimum-norm in new metric) fit all stored pairs exactly, and differ only off the span of the stored keys (different crosstalk onto new keys).

This is the congruence update $\Phi\mapsto P^{\top}\Phi P$, $\mu\mapsto \mu P$ used by DPCR for class covariances and prototypes (literature review, section 3), written for a key-value memory. It is **not new**.

## 3. The chosen hard sub-problem

**Problem.** Estimate $T$ (or its action) from paired before/after keys of *current* inputs only, state what is identifiable about *old* keys that are never revisited, and detect when the estimate must not be used; with $O(d^2)$-sized persistent state, no replay, one pass, CPU.

### 3.1 Estimator (ridge toward the identity)

$$\hat T = \arg\min_T \lVert K' - TK\rVert_F^2 + \lambda\lVert T - I\rVert_F^2 = \big(K'K^\top+\lambda I\big)\big(KK^\top+\lambda I\big)^{-1} = I + D K^\top (KK^\top+\lambda I)^{-1}.$$
Dual form (push-through identity): $\hat T = I + D\,(K^\top K+\lambda I_n)^{-1}K^\top = I + UV^\top$ with $U = D\in\mathbb{R}^{d\times n}$, $V = K(K^\top K+\lambda I_n)^{-1}\in\mathbb{R}^{d\times n}$. Shrinking toward $I$ rather than $0$ encodes the prior "drift is small where unobserved". Inverse by Woodbury: $\hat T^{-1} = I - U(I_n+V^\top U)^{-1}V^\top$.

Define $P_\lambda = KK^\top(KK^\top+\lambda I)^{-1}$ (a soft projector onto the excited subspace, eigenvalues $s_i/(s_i+\lambda)\in[0,1)$ for the eigenvalues $s_i$ of $KK^\top$). Then $I-P_\lambda = \lambda(KK^\top+\lambda I)^{-1}$. This is exactly the OWM projector $I - A(A^\top A+\alpha I)^{-1}A^\top$ of Zeng et al. (2019) applied to the window: the "excited subspace" machinery of OWM and of drift estimation are the same algebra.

Running-statistics variant (bounded, no window buffer): $S_{KK}\leftarrow \mu S_{KK}+kk^\top$, $S_{K'K}\leftarrow \mu S_{K'K}+k'k^\top$, $\hat T = (S_{K'K}+\lambda I)(S_{KK}+\lambda I)^{-1}$, $\mu\in(0,1]$. State $2d^2$ floats. Bias if the drift is nonstationary inside the window (earlier pairs reflect smaller accumulated drift); $\mu<1$ trades this against variance. **HYPOTHESIS** that $\mu$ can be set from the optimizer's step size.

### 3.2 Exact error identity (PROVEN)

For any invertible $\hat T$ and any drift map $\phi$ (linear or not), with $M' = M\hat T^{-1}$:
$$M'\phi(k) - Mk = M\hat T^{-1}\,r(k),\qquad r(k) = \phi(k) - \hat T k .$$
Retrieval error is the memory, amplified by $\hat T^{-1}$, times the **transport residual at the stored key**. With no correction ($M'=M$) the error is $M(\phi(k)-k)$.

For exactly linear noise-free drift $\phi = T$, $K'=TK$: $\hat T = I + (T-I)P_\lambda$, hence
$$T-\hat T = (T-I)(I-P_\lambda),\qquad r(k) = \lambda\,(T-I)(KK^\top+\lambda I)^{-1}k .$$
The residual is zero for $k$ in the excited subspace ($s_i\gg\lambda$) and approaches the full drift $(T-I)k$ for $k$ in directions the window does not excite ($s_i\ll\lambda$). With one batch, $\operatorname{rank}K\le n<d$, so most directions are unexcited.

### 3.3 Misspecification and noise (PROVEN, elementary)

Write $\phi(k) = Tk+\rho(k)$ where $T$ is the best linear approximation and $\Xi = [\rho(k_1),\dots,\rho(k_n)]$. Then
$$r(k) = \rho(k) + (T-I)(I-P_\lambda)k - \Xi K^\top(KK^\top+\lambda I)^{-1}k,\qquad \lVert K^\top(KK^\top+\lambda I)^{-1}\rVert \le \frac{1}{2\sqrt\lambda}.$$
(The singular values of $K^\top(KK^\top+\lambda I)^{-1}$ are $\sqrt{s}/(s+\lambda)\le 1/(2\sqrt\lambda)$, maximum at $s=\lambda$.) If $\lVert\rho(k)\rVert\le\eta\lVert k\rVert$ on the trust region, $\lVert T-I\rVert\le\varepsilon$ and $\lVert k\rVert\le1$:
$$\lVert r(k)\rVert \le \eta\Big(1+\frac{\sqrt n}{2\sqrt\lambda}\Big) + \varepsilon\,\lVert (I-P_\lambda)k\rVert .$$
So per-event error is controlled by three things: nonlinearity $\eta$, drift size $\varepsilon$ times the **uncovered energy** of the key, and the ridge-variance trade-off in $\lambda$.

### 3.4 Identifiability: what current inputs cannot tell you (PROVEN)

**Prop 4 (unexcited drift is unrecoverable).** Let the stored keys be orthonormal $e_1,\dots,e_N$ spanning $E$, with orthonormal values $v_i$ ($M = \sum_i v_ie_i^\top$). Let the window lie in $W\perp E$. Consider the family of linear, **orthogonal** (condition number 1) drifts $T_\sigma$ that fix $E^\perp$ and permute $e_i\mapsto e_{\sigma(i)}$, $\sigma\in S_N$. Every $T_\sigma$ yields the *same* window observations $K'=K$. Any transport rule, being a function of $M$ and the observations, outputs the same $M'$ for all $\sigma$. For $\sigma$ uniform,
$$\mathbb{E}_\sigma\sum_i \lVert M' e_{\sigma(i)} - v_i\rVert^2 \;\ge\; N-1,$$
while $M'=0$ gives $N$. Proof: with $u_j=M'e_j$, $s=\sum v_i$ ($\lVert s\rVert^2=N$), the left side is $\frac1N\sum_j\sum_i\lVert u_j-v_i\rVert^2$ and $\sum_i\lVert u-v_i\rVert^2 = N\lVert u-s/N\rVert^2 + N - 1\ge N-1$. Equality holds for $u_j=s/N$: the best rule returns the mean value, i.e. **retains nothing but the mean**.

Consequences. (i) No estimator, however sophisticated, transports memory along directions that the current stream never excites; this is a statement about information, not conditioning. (ii) The "trust the identity" fallback $M'=M$ has expected squared error $2(N-1)$ in this family, **twice the error of abstaining** ($N-1$ for returning the constant mean; or $N$ for returning zero). Where drift in unexcited directions is large, shrinking uncovered memory mass toward zero (abstention) beats assuming no drift. (iii) What rescues the identity prior is a *bound on drift magnitude* independent of the data, such as $\lVert\phi(k)-k\rVert\le\delta$ from the known parameter change and an encoder Lipschitz bound; then error $\le\lVert M\rVert\,\delta$ for $M'=M$ (PROVEN, trivial). The size of $\delta$ relative to key separation decides which regime applies. That relative size is measurable and is the central empirical quantity (section 8).

### 3.5 Exact preservation under arbitrary drift is impossible, even with unbounded memory (PROVEN)

**Prop 5 (collapse).** Suppose stored inputs $x_1,x_2$ have $f_s(x_1)\neq f_s(x_2)$, values $v_1\ne v_2$, and drift gives $f_t(x_1)=f_t(x_2)=q$ (e.g. $\phi(k)=\lvert k\rvert$ in one dimension collapses $\pm a$). Any memory read as $g(f_t(x))$, for any function $g$, any size, any algorithm, returns one output for both, so $\max_i\lVert g(q)-v_i\rVert\ge\lVert v_1-v_2\rVert/2$. Therefore exact preservation of all associations under arbitrary nonlinear drift is impossible for **every** reader that accesses the memory only through the current encoder. The only exact remedies change the read path (retain the old encoder, i.e. an anchor copy, and query through it), or replay raw inputs. Both are outside the stated constraints or reduce to the "frozen key" control.

The bounded-memory counting argument adds a second, independent limit: a linear memory of shape $m\times d$ stores at most $\min(m,d)$ independent associations exactly (rank); transport by invertible $\hat T$ preserves rank and so cannot raise capacity.

**Conclusion.** Exact preservation under arbitrary nonlinear drift with bounded memory: **no**. What is possible is bounded error under assumptions, section 4.

## 4. The approximate guarantee that is achievable

**Assumptions.**
- A1 (local near-linearity). Each consolidation event $j$ has $\phi_j(k)=T_jk+\rho_j(k)$ with $\lVert T_j-I\rVert\le\varepsilon_j$ and $\lVert\rho_j(k)\rVert\le\eta_j\lVert k\rVert$ on the keys of interest.
- A2 (keys bounded) $\lVert k\rVert\le1$.
- A3 (gate) The transport is applied only if $\sigma_{\min}(\hat T_j)\ge\sigma_0>0$.
- A4 (pairs are from the current stream through the same encoder pair that produced the drift), no label noise.
- A5 (observability of $\varepsilon_j$) $\varepsilon_j$ is bounded from the known parameter change; an isotropy proxy $\hat\varepsilon_j=\lVert\hat\Delta_j\rVert$ is an unverified stand-in.

**Per-event bound (PROVEN under A1-A3).** From section 3.2-3.3, with $M_j=M_{j-1}\hat T_j^{-1}$:
$$\lVert \text{err}_j(k)\rVert \le \frac{\lVert M_{j-1}\rVert}{\sigma_0}\Big[\eta_j\Big(1+\tfrac{\sqrt n}{2\sqrt\lambda}\Big)+\varepsilon_j\lVert (I-P_\lambda^{(j)})k\rVert\Big].$$
Also $\lVert M\hat T^{-1}\rVert\le\lVert M\rVert/\sigma_{\min}(\hat T)$ (stability: near-singular $\hat T$ amplifies the memory).

**Accumulation identity (PROVEN).** With true linear drifts $T_j$, estimates $\hat T_j$ and $G_j=\hat T_j^{-1}T_j-I$, the memory after $\tau$ events satisfies exactly
$$M_\tau\,T_\tau\cdots T_1 - M_0 \;=\; M_0\sum_{j=1}^{\tau}\big(\hat T_1^{-1}\cdots\hat T_{j-1}^{-1}\big)\,G_j\,\big(T_{j-1}\cdots T_1\big).$$
(Induction on $\tau$.) If every $\lVert\hat T_i^{-1}\rVert\lVert T_i\rVert\le 1+\eta_a$, then $\lVert\text{err}_\tau(k)\rVert\le\lVert M_0\rVert\lVert k\rVert\,\bar g\,\big((1+\eta_a)^\tau-1\big)/\eta_a$ with $\bar g=\max_j\lVert G_j\rVert$. Error therefore grows **linearly** in the number of events when steps are near-isometric ($\eta_a\to0$), and at worst exponentially otherwise. This is a clean, testable accumulation law.

**Honest limit on the gain.** Both the do-nothing error $M(\Phi_\tau-I)k$ and the transport error are first order in the step size; transport wins only through the factor $(I-P_\lambda)$, i.e. by the covered fraction of the drift. If the stream seldom excites the fact subspace, there is little to win (Prop 4).

**Trust-region statement (PROVEN, Taylor).** If $\phi\in C^2$ with $\lVert D^2\phi\rVert\le H$ and the window lies in a ball $B(c,R)$, the affine Taylor map at $c$ satisfies $\sup_{B(c,R)}\lVert\phi-\text{Aff}\rVert\le HR^2/2$, and a query at distance $r$ from $c$ obeys $\lVert\phi(k)-\text{Aff}(k)\rVert\le H(R+r)^2/2$. (Use an affine fit: augment keys with a constant coordinate, $d+1$.) **Counterexample:** $d=1$, $\phi(k)=k+ck^2$, window $\{1\}$: $\hat T=1+c$ is exact on the window; at $k=10$ the residual is $\phi(10)-\hat T\cdot10=90c$. Local transport is valid only near the keys used to estimate it. Whether an online-trained encoder's drift on the *stored* keys is inside such a trust region is **HYPOTHESIS** (section 8).

## 5. Reliability gate and fallback (design; DERIVED)

All quantities are computable from the window, $M$, and $O(d^2)$ state.
- **G1 invertibility and conditioning.** $\hat T=I+UV^\top$ has eigenvalues $1$ and those of $I_n+V^\top U$; require $\sigma_{\min}(\hat T)\ge\sigma_0$ and $\kappa(\hat T)\le\kappa_{\max}$ (cost $O(dn^2)$).
- **G2 fit reliability (nonlinearity detector).** Residual $R=D(I-H)=\lambda D(K^\top K+\lambda I)^{-1}$ with hat matrix $H=K^\top K(K^\top K+\lambda I)^{-1}$; the exact ridge LOO residuals are $R_{:i}/(1-H_{ii})$. Gate on $\rho_{\text{LOO}}=\lVert R_{\text{LOO}}\rVert_F/\lVert D\rVert_F$. Large $\rho_{\text{LOO}}$ means the window does not obey one linear map (nonlinear drift, or collapse).
- **G3 coverage of memory mass.** $c_M=\lVert M(I-P_\lambda)\rVert_F^2/\lVert M\rVert_F^2$ is the share of memory energy acting on directions the window does not excite; computed from $M$ alone (no stored keys): the row space of $M$ lies in the span of the stored keys.
- **G4 drift ledger (per-association reliability at read time).** Keep $L\in\mathbb{R}^{d\times d}$ (PSD, current coordinates) with $L\leftarrow \hat T^{-\top}\big[L+\hat\varepsilon^{2}(I-P_\lambda)^{\top}(I-P_\lambda)\big]\hat T^{-1}$ at each applied transport, and $L\leftarrow L+\hat\varepsilon^{2}I$ at a rejected one (the whole drift is unaccounted). A query key $k$ gets $\hat u(k)^2=k^\top Lk$, a conservative (it includes events before an item was written) first-order proxy for accumulated uncovered drift. Abstain if $\lVert M\rVert\,\hat u(k)>u_{\max}$ relative to the read margin. This is the "do not trust the transport for that association" fallback: per query, not per window. Status: derived by transporting the quadratic form $k^\top[\cdot]k$ through the congruence $\hat T$; the *form of the bound* (sum of per-event uncovered terms) is DERIVED from section 4, but whether $\hat u$ ranks failing facts correctly is a **HYPOTHESIS** (primary diagnostic, section 8).
- **Fallbacks.** Rejected window: keep $M$ unchanged and charge the ledger (identity prior with a recorded debt); if the debt is large, shrink or abstain, because under Prop 4 the identity prior can be worse than abstaining.

## 6. Complexity and persistent state

| Item | Shape | Cost |
|---|---|---|
| Memory $M$ (or $A$) | $m\times d$ | persistent |
| Key covariance $C$ (or its FD sketch $B$) | $d\times d$ (or $r\times d$) | persistent |
| Ledger $L$ | $d\times d$ | persistent |
| Running cross-statistics $S_{KK},S_{K'K}$ | $2\times d\times d$ | persistent (reset per anchor) |
| Anchor encoder copy (anchored mode only) | $\lvert\theta\rvert$ | persistent, constant |
| Estimate $\hat T$ (dual form) | $d\times d$ as $UV^\top$, rank $n$ | $O(dn^2+d^2n)$ per event |
| Transport of $M$ (Woodbury) | | $O(dn(m+n))$ per event |
| Transport of $(A,C)$ | | $O(d^2n)$ via Woodbury; $O(d^3)$ direct |
| RLS write | | $O(d^2)$ per fact |

Persistent state is $O(md+d^2+\lvert\theta\rvert)$, independent of the number of facts $N$ and of the number of drift events $\tau$. No raw input is stored. FD sketch (Liberty 2013; Ghashami et al. 2016): $B^\top B\preceq C$ and $C-B^\top B\preceq \frac{\lVert K_0-K_{0,k}\rVert_F^2}{r-k}I$; congruence preserves the Loewner order, so $T B^{\top}BT^{\top}$ sketches $TCT^\top$ with error at most $\lVert T\rVert^2$ times the original bound (PROVEN from the cited theorem; the sketch trades $d^2$ for $rd$).

Numerical stability: use $\lambda>0$ always; require $\sigma_0$ and $\kappa_{\max}$ gates (G1); compute the Woodbury core $I_n+V^\top U$ in the $n\times n$ space; keep keys normalized; accumulate $S$ in at least `f32` with symmetric re-enforcement, and treat a failed Cholesky of $C+\lambda I$ as a gate failure. Classical risk: RLS with forgetting factor loses information (covariance windup) when directions are not persistently excited; directional-forgetting variants (for example SIFt-RLS) keep the covariance bounded above and below; the ridge toward $I$ and a floor on $\lambda$ play that role here.

Capacity: independent of drift; at most $\min(m,d)$ exact associations; beyond that, crosstalk. After transport, crosstalk between two stored items scales by at most $\kappa(\hat T)$ (angles are distorted by the conditioning).

## 7. What this does not claim

- No transport of a nonlinear encoder's drift is claimed exact; Prop 4 and Prop 5 show where it provably fails.
- Per-item nonlinear transport (as in the prototype methods SDC and ADC) is not available for a superposed memory, because the stored keys are not separately recoverable from $M$; only a global (or subspace-wise) map can act.
- Composition of corrected memories (for example summing memories with different drift histories, or chaining facts across a transported memory) is **not** treated here. Section 3.2 gives the single-memory error; interference between independently transported memories is left open and is out of the smallest experiment's decision-bearing scope.
- The ZeroLM mainline today feeds memory from a frozen hashed-trigram key encoder (`memory-encode`), which has zero drift by construction. DCNM becomes relevant only if memory keys come from a learned, updating encoder (the "reading into memory" milestone). Any mainline use would also have to run on the neural ISA (ISA operations have no linear solve; a small matrix inverse would be built from matrix products, for example by Newton-Schulz iteration) and would add weight groups that must be declared in `model/specs/weight-groups.json` first.

## 8. Proven properties versus hypotheses to test

| # | Statement | Status |
|---|---|---|
| 1 | $M T^{-1}$ undoes linear invertible drift | PROVEN, trivial, not a contribution |
| 2 | Transport of $(A,C)$ equals the exact ridge memory on re-encoded keys; $\lambda>0$ differs from $M_\lambda T^{-1}$ unless $T$ orthogonal | PROVEN |
| 3 | Error identity $M\hat T^{-1}r(k)$; closed form $r(k)=\lambda(T-I)(KK^\top+\lambda I)^{-1}k$ for linear drift | PROVEN |
| 4 | Ridge noise/misspecification term bounded by $\lVert\Xi\rVert/(2\sqrt\lambda)$ | PROVEN |
| 5 | Unexcited drift is unrecoverable; identity prior can be twice as bad as abstaining | PROVEN (family of orthogonal drifts) |
| 6 | Exact preservation under arbitrary nonlinear drift impossible even with unbounded memory (collapse) | PROVEN |
| 7 | Accumulation identity; linear growth for near-isometric steps | PROVEN |
| 8 | Taylor trust-region error $H(R+r)^2/2$; counterexample $k+ck^2$ | PROVEN |
| 9 | FD sketch transports by congruence | PROVEN from cited theorem |
| H1 | Per-event drift of an online-trained small encoder is near-linear on the key space ($\eta_j$ small) | HYPOTHESIS |
| H2 | Drift on stored keys lies in directions the current stream excites ($c_M$ small) under realistic streams | HYPOTHESIS |
| H3 | Measured accumulation follows the linear-in-$\tau$ law of Prop 7 | HYPOTHESIS |
| H4 | The ledger $\hat u(k)$ and gates G1-G3 rank the facts that fail (calibration, AUROC) | HYPOTHESIS |
| H5 | At matched encoder plasticity, transport retains more than no correction, and more than prevention (OWM/OGD-style constraint) | HYPOTHESIS |
| H6 | A frozen or anchored key encoder does not give the same retention *and* the same benefit from key learning (otherwise there is nothing to solve) | HYPOTHESIS; the stop rule of the pivot |
| H7 | The identity prior is safe when accumulated drift $\delta$ is small relative to key separation (slow-drift regime, as reported for training embeddings by XBM) | HYPOTHESIS; defines when DCNM is unnecessary |

## 9. References for this document (details in the literature review)
Zeng et al., OWM, Nature Machine Intelligence 2019; He et al., DPCR, ICML 2025; Gomez-Villa et al., LDC, ECCV 2024; Yu et al., SDC, CVPR 2020; Liberty, Frequent Directions, KDD 2013 and Ghashami et al., SIAM J. Comput. 2016; Wang et al., test-time regression, 2025; von Oswald et al., MesaNet, 2025; Meng et al., ROME/MEMIT, 2022; Wang et al., RLSEdit, IJCAI 2026.
