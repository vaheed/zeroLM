# DCNM literature review

Status: DRAFT, 2026-11-27. Purpose: decide whether Drift-Corrected Neural Memory (DCNM; definition in `docs/research-pivot-dcnm.md`) is a known mechanism, and where an unresolved sub-problem remains under the project's constraints (bounded state, no replay of raw observations, single pass, CPU only). Math is in `docs/dcnm-mathematics.md`.

DCNM as stated: a persistent linear associative memory $M$ (read $v=Mk$) whose keys come from an encoder that keeps learning; after drift $k_{\text{new}}=Tk$, transport $M_{\text{new}}=MT^{-1}$.

**How this was done.** Web search and page fetches on 2026-11-26/27. "Read" below says how far each source was actually opened: *full* (method text read), *abstract* (abstract or abstract page only), *snippet* (search result summary only), *memory* (a classic cited from background knowledge, not opened this session; verify before citing externally). Titles and arXiv identifiers were checked against the arXiv API or Crossref. Where a source was only a snippet, no technical claim beyond the snippet is made. Searches were bounded; "not found" means not found in this search, not proven absent.

## Verdict summary

| Question | Answer |
|---|---|
| Is $M_{\text{new}}=MT^{-1}$ new? | No. Elementary linear algebra. |
| Is "estimate $T$ by ridge least squares from before/after features of the current data, transport the stored statistics by congruence, solve ridge in closed form" new? | **No.** It is the core of DPCR (ICML 2025) and, in a linear-projector form, of LDC (ECCV 2024); the translation-only special case is SDC (CVPR 2020). |
| Is the projector algebra for "which directions does the current data excite" new? | No. It is OWM (2019) and the null-space family (GPM, Adam-NSCL, AlphaEdit). |
| Is the bounded-state online regression machinery new? | No. RLS, ridge, Frequent Directions. |
| Does a published method transport a *key-value associative memory matrix* across changes of an *online-learned* key encoder? | **Not found.** Fast-weight and test-time-memory models (Titans, DeltaNet, Mesa) avoid the problem by training the key projections offline and freezing them at inference. Closest are the prototype and classifier methods above. |
| Open sub-problems not solved by the closest method under the constraints | (a) A per-association reliability certificate for a *superposed* memory when the current stream does not excite the old keys' directions (identifiability); (b) the accumulation of transport error over many events with bounded statistics; (c) abstention versus identity fallback; (d) non-injective or strongly nonlinear drift, where no method can be exact. Section 17 states them. |

## 1. Orthogonal Gradient Descent (OGD)

OGD (Farajtabar, Azizan, Mott, Li; AISTATS 2020, PMLR 108:3762-3773; https://proceedings.mlr.press/v108/farajtabar20a.html; arXiv 1910.07104; *read: abstract and method/limitations text*) projects new-task gradients onto the orthogonal complement of stored gradients of the old tasks' outputs, so old outputs change to first order by nothing. It "does not require storing the previously learned data", but it stores gradient vectors (a subset, 200 per task on MNIST), and the paper itself says the storage "grows with the number of tasks". Its guarantee holds near the old optimum, using the old gradients as proxies.
**Verdict.** Prevention, not transport: it constrains how the network may change, whereas DCNM lets the encoder change and corrects the memory. OGD's growing basis violates the bounded-state constraint; OWM and GPM (sections 2-3) bound it. Overlap with DCNM: the projector algebra (see OWM); no overlap in mechanism.

## 2. Orthogonal Weight Modification (OWM)

Zeng, Chen, Cui, Yu, *Continual learning of context-dependent processing in neural networks*, Nature Machine Intelligence 1:364-372, 2019 (DOI 10.1038/s42256-019-0080-x; arXiv 1810.01256; *read: abstract and search summary*). The update is $\Delta W=\kappa P\,\Delta W_{BP}$ with $P$ the projector orthogonal to the inputs of earlier tasks, maintained by recursive least squares: $P=I-A(A^\top A+\alpha I)^{-1}A^\top=\alpha(AA^\top+\alpha I)^{-1}$.
**Verdict.** The soft projector $I-P_\lambda=\lambda(KK^\top+\lambda I)^{-1}$ in `docs/dcnm-mathematics.md` is this projector. OWM keeps old-input statistics (bounded, $O(d^2)$) and stops the layer from changing on them (prevention). Overlap is exact on the algebra, none on the idea of correcting drift after it happens. OWM is the natural *prevention* baseline arm.

## 3. Gradient projection, null-space training, null-space editing

Gradient Projection Memory (Saha, Garg, Roy; arXiv 2103.09762; *abstract*) stores SVD bases of each layer's activations per task and takes gradient steps orthogonal to them. Adam-NSCL (Wang et al., *Training Networks in Null Space of Feature Covariance for Continual Learning*, arXiv 2103.07113, CVPR 2021 from *memory*; *abstract*) projects updates into the approximate null space of the uncentered feature covariance, which is "incrementally computed". AlphaEdit (Fang et al., ICLR 2025; arXiv 2410.02355; *abstract*) projects a model-editing perturbation onto the null space of preserved knowledge and proves that outputs for preserved knowledge stay unchanged.
**Verdict.** Same family as OWM: bounded covariance or basis statistics, prevention by constraint. Their cost is plasticity: updates are confined to a shrinking subspace. Overlap with DCNM: the statistics and the subspace algebra; DCNM's premise is that the encoder is allowed to change anywhere.

## 4. Semantic drift compensation for prototypes and classifiers (closest family)

SDC (Yu et al., *Semantic Drift Compensation for Class-Incremental Learning*, CVPR 2020, pp. 6982-6991; arXiv 2004.00440; *abstract*) approximates the drift of old classes "based on the drift that is experienced by current task data" and shifts old class prototypes by a locally weighted average of the current samples' per-sample drift; per LDC's analysis its assumption is that drift is locally a translation. ADC (Goswami et al., CVPR 2024; arXiv 2405.19074; *abstract*) perturbs current samples adversarially until they sit near old prototypes in the old model's embedding, then measures their drift. LDC (Gomez-Villa et al., *Exemplar-free Continual Representation Learning via Learnable Drift Compensation*, ECCV 2024; arXiv 2407.08536; *read: method and ablation text*) trains a forward projector from old to new features by mean squared error on current-task data only and applies it to the stored prototypes; its ablation reports that a single linear layer worked better than linear+ReLU or an MLP (45.4 against 43.7 and 41.2 on one setting). DPCR (He et al., *Semantic Shift Estimation via Dual-Projection and Classifier Reconstruction for Exemplar-Free Class-Incremental Learning*, ICML 2025, PMLR 267; arXiv 2503.05423; *read: method text via the arXiv HTML*) fits a single linear map $\hat P=(X^\top X+\epsilon I)^{-1}X^\top X'$ from old-backbone to new-backbone features on the current task (ridge, Eq. 5), adds a class-wise row-space projection $P_c=PU^r(U^r)^\top$, stores per-class uncentered covariance $\Phi_c\in\mathbb{R}^{d\times d}$ and prototype $\mu_c$, transports them by $\hat\Phi=P_c^\top\Phi P_c$ and $\hat\mu=\mu P_c$, and reconstructs a ridge classifier in closed form from the transported statistics.
**Verdict: this is the closest existing method, and DCNM's core is mathematically equivalent to it.** With key-value pairs in place of class prototypes, $(A,C)=(VK^\top,KK^\top)$ in place of (prototype, covariance), and $M=A(C+\lambda I)^{-1}$ in place of the ridge classifier, the transport of the stored statistics is DPCR's, and the closed-form memory equals the ridge memory on re-encoded keys (Prop 1 of the math doc). DPCR explicitly does not treat nonlinear drift and gives no guarantee on the transfer error from current data to old-class embeddings (as read in its limitations); its storage is $d^2+d$ per class, which is unbounded in the number of classes. LDC's linear-beats-MLP result is a datapoint that, in that setting, the linear transport is enough; it is also one setting.

## 5. Embedding-space alignment, backward compatibility, and drift adapters

Backward-compatible representation learning (Shen et al., CVPR 2020; arXiv 2003.11942; *abstract not opened beyond the listing*) trains the new model so its embeddings remain comparable with the old gallery. Drift-Adapter (Vejendla, EMNLP 2025 main, arXiv 2509.23471; *snippet*) maps new-model query embeddings into the legacy space with a small adapter (orthogonal Procrustes, low-rank affine, or a residual MLP) trained on a small paired sample, reporting 95-99% recall recovery; relative representations (Moschella et al., arXiv 2209.15430; *listing only*) argue for angle-based invariance to transformations of the latent space. Procrustes alignment (Schönemann, Psychometrika 1966; *memory*) is the classical orthogonal case.
**Verdict.** Known and used: a linear (often orthogonal) map between embedding versions, estimated from paired samples, applied to a stored index. DCNM's transport is the same operation applied to a weight matrix instead of an index; mathematically identical to a linear adapter in the retrieval world.

## 6. Representational drift and drift-tolerant decoding (neuroscience, brain-computer interfaces)

Degenhart et al. (*Stabilization of a brain-computer interface via the alignment of low-dimensional spaces of neural activity*, Nature Biomedical Engineering 4:672-685, 2020; DOI 10.1038/s41551-020-0542-9; *snippet and abstract*) align each day's low-dimensional neural manifold to a reference so a fixed decoder keeps working. Rule et al. (*Stable task information from an unstable neural population*, eLife 9:e51121, 2020; *snippet and abstract*) find that drift is constrained far above chance, so a linear readout can follow it, but that a significant component of drift degrades a fixed readout, so some compensation by plasticity is needed.
**Verdict.** The same problem (a fixed readout over a drifting code, solved by alignment on current data) is an established idea in neuroscience and neural engineering. Not novel; supports the plausibility of low-dimensional, mostly-linear drift and also the finding that part of drift is not in a null space (compare Prop 4).

## 7. Slow-drift and stability-by-design alternatives

XBM (Wang et al., *Cross-Batch Memory for Embedding Learning*, CVPR 2020; arXiv 1912.06798; *snippet*) observes that embeddings change only slowly during training, so stale stored embeddings remain usable without correction; MoCo-style momentum encoders keep drift small by construction. The analytic-learning line (ACIL, Zhuang et al., NeurIPS 2022, arXiv 2205.14922, *abstract*; F-OAL, arXiv 2403.15751, *listing*; RanPAC, McDonnell et al., arXiv 2307.02251, *listing*; streaming LDA, Hayes and Kanan, arXiv 1909.01520, *listing*) obtains exactly the joint-training solution by recursive least squares, but over a **frozen** backbone: it removes drift by not learning the representation.
**Verdict.** These are the control that decides whether a drift problem exists at all: if drift is slow, or if keys can be frozen, DCNM solves nothing. They are also the reason the experiment needs a frozen-key and a no-correction arm (Section 17 and the design skeleton).

## 8. Recursive least squares, online ridge regression, directional forgetting

RLS (Plackett 1950; Haykin, *Adaptive Filter Theory*; Ljung and Söderström, *Theory and Practice of Recursive Identification*; all *memory*) is the exact online solution of ridge regression by a Sherman-Morrison update at $O(d^2)$ per sample, and was applied to continual learning in OWM and ACIL. A known failure is covariance windup when directions are not persistently excited under a forgetting factor; remedies are directional forgetting and resetting (SIFt-RLS, Lai et al., arXiv 2404.10844, *snippet*, proves upper and lower covariance bounds without persistent excitation). RLS has also reached language-model editing (RLSEdit, Wang et al., *Beyond Hard Writes and Rigid Preservation: Soft Recursive Least-Squares for Lifelong LLM Editing*, IJCAI 2026, arXiv 2601.15686; *snippet*; Woodbury recursion with per-edit cost independent of history, 10,000 edits) and Analytic Subspace Routing (Tong et al., arXiv 2503.13575; *snippet*).
**Verdict.** DCNM's estimator and its memory are RLS; the excitation problem is the textbook identifiability condition. No novelty. The windup literature supplies the stability conditions used in the math doc.

## 9. Online covariance estimation, streaming subspace tracking, matrix sketches

Frequent Directions (Liberty, KDD 2013, arXiv 1206.0594 *title verified*; Ghashami, Liberty, Phillips, Woodruff, *Frequent Directions: Simple and Deterministic Matrix Sketching*, SIAM J. Comput. 2016, arXiv 1501.01711 *title verified*) keeps an $\ell\times d$ sketch $B$ with $0\preceq A^\top A-B^\top B$ and spectral error $\le\lVert A-A_k\rVert_F^2/(\ell-k)$. Incremental SVD (Brand 2002), Oja's rule (1982), GROUSE (Balzano et al., 2010) and PAST (Yang, 1995) track subspaces from streams (*all memory*).
**Verdict.** The tools for bounding $C=KK^\top$ exist; a sketch transports by congruence (math doc, section 6). Using them here is engineering.

## 10. Fast-weight and associative-memory sequence models

Fast weights (Schmidhuber 1992, Neural Computation 4(1), *memory*; Ba et al., *Using Fast Weights to Attend to the Recent Past*, NeurIPS 2016, arXiv 1610.06258, *title verified*); linear attention equals an outer-product fast-weight memory with a delta-rule variant (Schlag, Irie, Schmidhuber, ICML 2021, arXiv 2102.11174, *title verified; cited in the repo's novelty review*); modern Hopfield networks (Ramsauer et al., arXiv 2008.02217, *title verified*). Correlation matrix memories (Kohonen 1972; Anderson 1972; *memory*) are the origin of $v=Mk$.
**Verdict.** $M$ and its delta-rule write are the same mathematics (as `docs/novelty-analysis.md` already records). These models assume the key map is fixed or trained in an outer loop; see the next section.

## 11. Titans, Miras, Atlas, test-time training, DeltaNet, Gated DeltaNet, Mesa

Titans (Behrouz, Zhong, Mirrokni, *Titans: Learning to Memorize at Test Time*, NeurIPS 2025, arXiv 2501.00663; *read: abstract, introduction and memory-module section of the NeurIPS PDF*) updates a (deep) memory $M_t=M_{t-1}+S_t$, $S_t=\eta_tS_{t-1}-\theta_t\nabla\ell(M_{t-1};x_t)$ with $\ell=\lVert M(k_t)-v_t\rVert^2$, $k_t=x_tW_K$; it states that $W_K,W_V$ are outer-loop parameters ("hyperparameters" of the inner loss), i.e. **trained offline and fixed at test time**. Miras (arXiv 2504.13173; *abstract*) recasts sequence models as associative memories with an attentional-bias objective and retention (forgetting) regularization; ATLAS (arXiv 2505.23735; *title only*, a follow-up on optimally memorizing context at test time); test-time regression (Wang et al., arXiv 2501.12352; *abstract*) treats memorization as regression and shows linear attention, state-space models, fast-weight programmers and softmax attention as special cases defined by weights, function class, and test-time optimizer; TTT layers (Sun et al., arXiv 2407.04620, *title verified, in the repo review*); DeltaNet (Yang et al., arXiv 2406.06484); Gated DeltaNet (Yang, Kautz, Hatamizadeh, arXiv 2412.06464, in the repo review); Longhorn (Liu et al., arXiv 2407.14207, *title verified*); the Mesa layer and MesaNet (von Oswald et al., arXiv 2309.05858 and 2506.05233; *abstract of MesaNet*) solve the in-context ridge problem to optimality at each step with a conjugate-gradient solver, i.e. RLS-like exact memories at test time.
**Verdict.** These supply the memory mathematics, including the exactly-optimal ridge memory (Mesa). None treats drift of the key map: by design the key map is frozen when the memory runs. DCNM's setting (key encoder trained online while a persistent memory is held) is therefore outside what they assume, but the setting is a different *regime*, not a different mechanism. In the ZeroLM repository a gated-delta memory was tested in experiment GDM (closed, FAIL) on a different question (compiled versus random initialization); its registered metrics did not isolate transport of stored associations under key drift, and this review does not revisit it.

## 12. Model editing: ROME, MEMIT, EMMET, sequential editing and forgetting

ROME (Meng et al., *Locating and Editing Factual Associations in GPT*, NeurIPS 2022, arXiv 2202.05262) and MEMIT (*Mass-Editing Memory in a Transformer*, ICLR 2023, arXiv 2210.07229) treat an MLP down-projection as a linear associative memory (key-value view of feed-forward layers: Geva et al., arXiv 2012.14913), and solve a least-squares edit that preserves the old key covariance $C=\mathbb{E}[kk^\top]$ measured offline; EMMET (Gupta, Sajnani, Anumanchipalli, *A Unified Framework for Model Editing*, Findings of EMNLP 2024, https://aclanthology.org/2024.findings-emnlp.903/; *read: abstract page*) shows ROME and MEMIT share a "preservation-memorization objective" differing in an equality versus least-squares constraint. Gupta, Rao, Anumanchipalli (*Model Editing at Scale leads to Gradual and Catastrophic Forgetting*, arXiv 2401.07453, ACL 2024 Findings; *abstract page*) show sequential edits forget earlier ones, first gradually and then abruptly; AlphaEdit (section 3) and RLSEdit (section 8) are responses.
**Verdict.** The memory-as-weight-matrix, key-covariance, and least-squares edit view is exactly the algebra of `M = A(C+λI)^{-1}`. In these methods the *model's keys are fixed* while edits accumulate (the failure is interference between edits); they do not face an encoder that is itself updated. Overlap: algebra and the preserve-versus-memorize trade-off. The "catastrophic collapse after many edits" result is relevant evidence that bounded linear memory has a capacity cliff independent of drift.

## 13. Bayesian continual learning

EWC (Kirkpatrick et al., PNAS 2017, arXiv 1612.00796), Variational Continual Learning (Nguyen et al., ICLR 2018, arXiv 1710.10628), online structured Laplace (Ritter et al., NeurIPS 2018, arXiv 1805.07810; all *titles verified*), and MESU (Bonnet et al., *Bayesian continual learning and forgetting in neural networks*, Nature Communications 16:9614, 2025, DOI 10.1038/s41467-025-64601-w; *read: abstract via Crossref*), a Bayesian update that scales each parameter's learning by its uncertainty with no task boundaries, protect important parameters through a posterior or Fisher penalty. Their cost is memory proportional to the parameter count or sampling, and they constrain, rather than transport.
**Verdict.** Prevention family again. An uncertainty-based ledger for transport reliability (math doc, section 5, G4) is conceptually adjacent to these but is a different object (uncertainty about the *transport*, not about weights); no source here does it.

## 14. Compositional memory and systematic generalization

Davis, Katz, Gentili, Reggia (*Compositional memory in attractor neural networks with one-step learning*, Neural Networks 138, 2021, DOI 10.1016/j.neunet.2021.01.031; *read: title and journal via Crossref; abstract not opened*); Lake and Baroni (*Generalization without systematicity*, ICML 2018, arXiv 1711.00350, *title verified*); tensor product representations (Smolensky, Artificial Intelligence 46, 1990), holographic reduced representations (Plate, IEEE TNN 1995), and the classical systematicity argument (Fodor and Pylyshyn, Cognition 1988) (*memory*).
**Verdict.** Composition of corrected memories is not addressed by any source found as a *drift* problem. Superposition of bound representations is the standard mechanism for composition, and its interference limit (crosstalk grows with load) applies after transport unchanged. The repo's reasoning module already uses phasor binding (FHRR). DCNM's "compose" claim is untreated and left out of the smallest experiment's decision-bearing scope.

## 15. Not found / limits of this review

- No source found that combines (i) a superposed key-value memory matrix, (ii) an online-trained key encoder, and (iii) transport estimated from current-stream before/after pairs with a per-association reliability certificate. The closest, DPCR, is class-based with $O(d^2)$ per class.
- Not opened: ATLAS body, Miras body, F-OAL, Drift-Adapter body, Analytic Drift Resister (arXiv 2604.02633, snippet only), ASR body, SIFt-RLS body, RLSEdit body, UltraEdit.
- Citation details taken from memory (classics, venues of GPM, Adam-NSCL, EFC, ROME/MEMIT, Titans/TTT/Gated DeltaNet venues) should be rechecked before external use.

## 16. Per-family overlap with the DCNM proposal

| Mechanism family | Closest method | Overlap with DCNM as stated | Verdict |
|---|---|---|---|
| Drift compensation | DPCR; LDC; SDC; ADC | Estimate map on current data; transport stored statistics | Equivalent in core (linear transport of covariance and prototype, ridge fit) |
| Feature alignment | Procrustes; Drift-Adapter; BCI manifold alignment | Linear map between paired embeddings applied to stored index | Equivalent operation |
| OGD, OWM, GPM, Adam-NSCL, AlphaEdit | OWM | Same projector algebra; opposite strategy (prevent vs transport) | Algebra equal; mechanism different; needed as baseline |
| RLS, ridge, forgetting | RLS, SIFt-RLS, RLSEdit, Mesa | Memory update and estimator are RLS | Equivalent |
| Covariance, sketches | Frequent Directions | Bounded $C$ | Engineering |
| Fast weights, Titans, Miras, DeltaNet, Mesa | Mesa/Titans | Same associative memory; key map frozen | Same memory; drift not addressed (fixed key regime) |
| Model editing | MEMIT, AlphaEdit, RLSEdit | Same least-squares memory view; fixed keys | Same algebra; different problem (interference of edits) |
| Bayesian CL | MESU, EWC, VCL | Constrain weights | Different mechanism |
| Compositional memory | Smolensky; Plate; Davis | Superposition | Not addressed by DCNM |

## 17. The residual research gap

After removing what is known, what remains is not a mechanism but a **characterization under the project's constraints**, none of which the closest method (DPCR) provides:

1. **Per-association reliability for a superposed memory.** DPCR transports class statistics but gives no guarantee on the transfer from current data to old-class embeddings. For a superposed memory the question is which *stored associations* are inside the excited subspace. The math doc gives a computable certificate (G3, G4) and proves a lower bound (Prop 4) showing why a certificate is needed: with unexcited drift, the identity fallback is twice as bad as abstaining.
2. **Bounded auxiliary state.** DPCR's $d^2$ per class is unbounded in the number of classes; a single $d\times d$ statistic (or an FD sketch) shared by all facts is bounded in the number of facts.
3. **Accumulation over many events** with bounded statistics: an exact identity and a linear-in-$\tau$ law for near-isometric steps (Prop 7); empirical agreement not known.
4. **Nonlinear and non-injective drift:** no method is exact (Prop 5); what is open is *detection* (G2) and graceful degradation, not correction.
5. **Demand:** whether any drift problem exists once a frozen or slowly-moving key encoder is the alternative (section 7). If not, there is nothing to solve.

Honest reading: items 1-3 are a modest, testable extension of known linear-transport machinery to a different object; they are not a new transport mechanism. The experimental design (`docs/dcnm-experimental-design.md`) is built so the demand question (5) is answered first.
