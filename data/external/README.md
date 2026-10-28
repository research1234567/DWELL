# External data (not distributed)

The real-data scripts expect `real_12m_continuous_panel.csv`. Its source and redistribution terms have not been verified, so it is excluded from the public repository. Do not substitute synthetic data and describe it as the real panel.

If independently obtained with the necessary rights, place the CSV in this directory. The runner copies it into its ignored working layout. Required columns are `hour`, `block`, `asset`, `cash`, `debt`, `total_aToken`, and `utilization`. The loader interprets `hour` relative to 2023-03-01; this is a code convention, not an independently verified data-collection date.
