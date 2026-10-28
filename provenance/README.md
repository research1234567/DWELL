# Provenance

`source_manifest.json` maps curated computational files and frozen synthetic outputs to their paths in the source package and records the published SHA-256 of each file. This is an integrity record, not evidence that every experiment was independently replicated.

The only change within copied computational source is removal of a machine-specific fallback path from `src/legacy/real_panel.py`. `scripts/run.py`, the smoke tests, dependency list and repository documentation were added for this layout. No simulator parameters, selection rules, seeds or stored numerical outcomes were changed.

`SHA256SUMS.txt` records all publication files other than itself. Generated outputs stay in ignored `work/`. Manuscript sources, compiled paper PDFs, revision correspondence, machine-specific orchestration scripts, copied literature pages and unverified external data are not part of this publication.
