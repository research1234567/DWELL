# Protocol: competitor baselines (locked before any sweep, 2026-10-28)

Question: do decaying-step online conformal (Angelopoulos, Barber, Bates 2024, arXiv:2402.01139) and conformal PID
(Angelopoulos, Candes, Tibshirani, NeurIPS 2023, arXiv:2307.16895) match or beat the capped ensemble in the same
endogenous-feedback simulator? Their primary-paper updates were implemented from the papers' stated equations; the
authors' repository (github.com/aangelopoulos/conformal-time-series) was located but its source files could not be fetched,
so code-level fidelity to it is NOT verified.

Simulator: identical to env_bench.run (T=8000, 2000-step calm/reactive blocks, W=500 initial window, RNG 1000+seed,
same draw order). Score is the outcome y; threshold-space methods are unprojected, deployed threshold clipped to [-1,1]
(empty/full set). Initial threshold: empirical 0.9 quantile of the initial window (same information all methods get).

Methods
 A decay_thr: q+=c t^-0.6 (err-alpha)   (eps=0.1 as in the paper's experiments)
 B decay_lvl: same decaying step on the level a_t with window quantile (matches the paper's level-form arms)
 C qt_thr: quantile tracking, eta fixed; plus untuned default eta=0.1*trailing-max(window) (paper's heuristic)
 D pid_thr: window-quantile scorecaster + tan integrator K_I tan(S_t log t/(t Csat)), no D/derivative term, no regression scorecaster.
Budgets
 default : no tuning (C only)
 pooled  : ONE hyper-parameter setting per method, chosen to minimise mean dev pinball over all four regimes (regime-blind)
 perreg  : best setting per regime on dev seeds (as informed about the regime as the scenario cap; favourable to baselines)
Dev seeds 5000-5009 (logit-normal), 5000-5004 (heavy-tailed). No tuning on evaluation seeds.
Evaluation seeds 0-49 (logit-normal), 0-19 (heavy-tailed): the same seeds as the stored guard/fixed/DtACI arrays.
Selection criterion: mean pinball (same metric as headline). Everything unfavourable is reported.
Grids: decay_thr c in {.1,.2,.5,1,2,5,10}; decay_lvl c in {.1,.2,.5,1,2,5};
 qt eta in {.0005,.001,.002,.005,.01,.02,.05,.1}; pid K_I in {.05,.2,1} x Csat in {heuristic(delta=.01)=0.566, 2.0}.
