# DWELL

Code and stored simulation results for **DWELL: Delay and Window Effects on the Stability of Quantile Tracking under Endogenous Feedback**.

This repository studies delayed endogenous feedback in quantile tracking. The stability analysis concerns deterministic threshold and population-window models. It does not establish a stability guarantee for empirical rolling quantiles or the full adaptive ensemble. Stability boundaries are not loss-optimal step sizes. Unfavorable simulation outcomes are retained.

## Repository layout

```text
src/
  legacy/        Original simulation, theory and calibration scripts
  baselines/     Competitive baselines and non-oracle cap experiments
  window/        Window-boundary calculations and extended experiments
  transfer/      Threshold-to-window transfer simulations and checks
  population/    Numerical population-model checks
scripts/run.py   Isolated runner for the archived scripts
protocols/       Historical experiment specifications
results/
  benchmarks/    Frozen synthetic benchmark JSON, caches and selections
  transfer/      Frozen transfer summaries and cell-level CSV
  population/    Stored numerical checks
data/
  synthetic/transfer/  Six compressed simulation-array files
  external/            Local-only external input (not distributed)
provenance/      File mapping and SHA-256 checksums
tests/          Fast deterministic checks
```

The `legacy` label indicates source lineage, not that every script supports a current headline claim. Historical protocols retain their original paths and seed counts; the later window protocol describes the heavy-tailed extension to 50 evaluation seeds. They are local records, not independent preregistrations.

## Setup and quick checks

Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

## Reproduce selected analyses

Run from the repository root:

```bash
# Summarize the stored window-cap and ablation runs.
OPENBLAS_NUM_THREADS=1 python scripts/run.py window w1_analyze.py
# Summarize the heavy-tailed experiments over 50 evaluation seeds.
OPENBLAS_NUM_THREADS=1 python scripts/run.py window g2_analyze50.py
# Check transfer calculations against a scalar reference simulation.
OPENBLAS_NUM_THREADS=1 python scripts/run.py transfer audit.py
```

The runner creates an ignored `work/` directory with the relative layout expected by the archived scripts. Outputs go there, never into the frozen `results/` or `data/synthetic/` directories. Existing working caches are reused. Remove `work/` to start from the frozen snapshot again.

To rerun the transfer simulation rather than summarize stored results:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/run.py transfer run_transfer.py --smoke
OPENBLAS_NUM_THREADS=1 python scripts/run.py transfer run_transfer.py
```

The smoke run is a smaller diagnostic, not a reported experiment. The full run uses 64 replicate columns, 4,000 steps and six scenario/window cells. Other runners have script-specific arguments; read their source and the matching protocol before starting long runs. Not every historical experiment was rerun when this repository was organized.

## Data and provenance

Synthetic transfer arrays are documented in `data/synthetic/README.md`. Frozen file origins and hashes are in `provenance/source_manifest.json`; `provenance/SHA256SUMS.txt` covers the publication snapshot. Computational sources and frozen outcomes are preserved byte-for-byte except for removal of one machine-specific fallback path in the real-data loader.

The external Aave panel is not distributed because its source and redistribution terms have not been verified. Real-data calibration and replay cannot be reproduced from this repository alone. Manuscript files, LaTeX dependencies, compiled paper PDFs and private revision records are excluded.

No software or dataset license is added by this upload. Do not infer third-party redistribution rights from repository visibility.
