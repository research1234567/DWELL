# Synthetic transfer data

`transfer/empirical_D_W.npz` contains generated simulation arrays, where `D` is the response delay (1, 6 or 12) and `W` is the rolling window (100 or 500). The six files are preserved from the reported transfer run.

Load with `numpy.load(path, allow_pickle=False)`.

- Ratio axis: `[0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]`, relative to the threshold-model critical gain.
- Perturbation axis: initial level offsets `[0, -0.002, 0.002, -0.01, 0.01]`.
- Replicate axis: 64 columns generated from a master RNG initialized at 47001. These are not 64 independently initialized RNG objects. Innovations are shared across arms within each replicate.
- `mse`, `mean_q`, `pinball`, `outside_fraction`, `raw_lag1`, `variance`: shape `(7, 5, 64)`, summarized over steps 1000 through 3999 (zero-based). `raw_lag1` is an uncentered lag-one ratio around the population equilibrium, not Pearson correlation.
- `mean_q_time`: shape `(4000, 7, 5)`, thresholds averaged across replicate columns at each time.
- `paired_small`, `paired_large`: shape `(7, 64)`, time-averaged positive-minus-negative threshold differences across all 4000 steps. They are not full trajectories or derivatives normalized by perturbation size.

The cell-level CSV in `results/transfer/` summarizes the unperturbed arm. Scenario parameters, metric uncertainty and paired-response summaries are in `results/transfer/results.json`. Full per-replicate trajectories are not stored; the runner can regenerate them.
