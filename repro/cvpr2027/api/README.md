# Frontier API evaluation core

This is a source-only extraction from the original frontier experiment, not a turnkey reproduction or an approval to spend money. It contains no dataset, private sample IDs, images, raw responses, credentials, original config, execution launcher, server setup or upload logic. The default transport cannot make network calls. The historical $75 cap is not carried forward.

Run from the repository root with Python 3.10+ and Pillow available:

```sh
python -m unittest repro.cvpr2027.api.test_api_runner -v
```

Import `repro.cvpr2027.api.api_runner`. The shared parser comes from root `egoown_cvpr.core`. `make_payload(root, model, item)` checks image bytes and the frozen prompt fingerprint; `infer(..., transport=callable)` supports an explicitly supplied transport for a separately authorized implementation. No transport is included. The transport signature matches `request(key, endpoint, payload=None, timeout=...)`; it returns provider JSON or raises an HTTP/transport exception. Importing or testing this package performs no network operation. `retryable` is historical response metadata, not an instruction to retry.

The caller supplies licensed rendered PNGs and a frozen item with `id`, `condition`, `prompt`, `images` (suite-relative paths), `image_sha256`, `version`, `input_sha256`, ordered `options`, and `gt_label`. Ground truth is response bookkeeping and never sent in the payload. A model record needs `id`, `key`, `canonical_slug`, `params`, and `expected_provider`. `model-parameters.template.json` records the original two public model IDs and their pinned parameters; execution is disabled and no budget is approved. Model availability is not guaranteed.

`pilot_report` requires model/condition lists, `n`, the original pilot ID set and first-attempt records. It checks technical validity, never accuracy. Frozen-input SHA uses the original sorted JSON serialization. Invalid outputs remain unparsed under the shared parser. Supplying changed prompts, image order, generation parameters or model versions defines a different experiment.

Excluded: original Runner, CLI, account/key management, backup/checkpoint deployment, source assets and private-data-dependent `test_frontier.py`. Nine synthetic tests from `test_api_runner.py` remain; two Runner budget tests are excluded. Recreating the original 200 or final common189 cohort needs separately licensed inputs and exact manifests; this package supplies neither.

`code-provenance-fragment.json` records original source and output SHA-256 with modifications. Original paths are relative to the private `data-work` root; they are provenance labels, not runtime dependencies.
