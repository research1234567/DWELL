# Protocol 2: non-oracle step caps (locked 2026-10-28 before any result of these arms exists)
These are two distinct empirical approaches. Neither proves the oracle stability theory, neither identifies rho/d from real data.
The earlier single-spike probing identified-cap attempt (AUDIT.md) gave no gain; it is not rerun and is reported as such.
Simulator: adaptive_cap_sim.run (copy of env_bench.run; deployed-error updating, same experts, RNG, metrics).
 (1) LORO constant cap: grid {0.003,0.006,0.012,0.025,0.05,0.2}; for held-out regime R pick the cap with the lowest mean dev
     pinball averaged over the OTHER three logit-normal regimes (dev seeds 5000-5005); evaluate on R, seeds 0-49.
 (2) Backoff(kappa=2,kappa_lo=1.3,down=.7,up=1.05,every=50,hor=100,cmax=.2,cmin=.002): fixed a priori, no tuning.
     Evidence: dev seeds 5000-5005 reported first; evaluation seeds 0-49 reported regardless of outcome. A failing dev result is reported, not retuned.
Comparators on the same seeds: stored EG scenario / universal / uncapped deployed arms, fixed steps, baselines of Protocol 1.
