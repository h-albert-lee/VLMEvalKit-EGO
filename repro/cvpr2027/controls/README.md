# Visual-input controls: portable analysis components

These are extracted components, not the exact historical dispatcher or a complete rerun bundle. No licensed images, item IDs, review records, predictions, or model weights are included.

- `selection.py`: unchanged function bodies from the historical supplement preparation. Selects 100 T3 and 100 T4 items, one per video/session, excluding original panel/demonstration groups and changed target names. Requires final and prior metadata supplied by the caller. This is a subset of the existing benchmark, not an unseen test set.
- `input_variants.py`: new thin wrapper for the historical exact text/image changes. Requires a sparse query with its image hashes. Repetition changes the view instruction and repeats the last image; text-only removes images and announces their absence; name-hidden changes only the target-name prefix. Rendering, parsing and scoring remain in `egoown_cvpr`.
- `analysis.py`: unchanged `outcome` and `paired_summary` function bodies. Keeps ERROR/UNPARSED wrong and preserves pairs. The 5,000-draw intervals are pointwise, not multiplicity-corrected. Legacy keys use `sparse` for the baseline. One item per video/session is required for item bootstrap to equal cluster bootstrap.

From the repository root, run `PYTHONPATH=. python -m unittest discover -s repro/cvpr2027/controls -p 'test_*.py'`. Tests use synthetic records only. Private-bundle completeness tests are intentionally excluded.

All restricted-view scores use the original full-window references. Text-only abstention is not proof that language shortcuts are absent; chronological-versus-repeat changes available observations and instructions and does not isolate temporal-order reasoning.
