# Label-adaptation offline core

This source-only package preserves frozen message construction, input identity, API receipt classification, pinned GPU encoding, first-attempt bookkeeping and scoring. It does not include licensed images, manifests, fixed human examples, predictions, credentials, private review records, model weights or the original server/API launcher. It is not turnkey exact reproduction. Execution is disabled: no model dispatch or network transport is supplied, and no new budget is approved.

From the repository root, with Python 3.10+ and Pillow:

```sh
python -m unittest repro.cvpr2027.label_adaptation.test_gpu_backend repro.cvpr2027.label_adaptation.test_runtime -v
```

Import modules through `repro.cvpr2027.label_adaptation`. They use the repository-root `egoown_cvpr.core` parser and metrics; no vendored package is expected. The GPU tests use a fake processor and synthetic PNGs, requiring no torch, weights or GPU.

- `messages.make_request` preserves B0 removal of the exact rules block, B1 baseline rules and B2 fixed four-example dialogue. The caller supplies a validated model record (`key`, `id`, `revision`, `provider`, `backend`, `parameters`, `wire_order`), frozen query (`baseline_prompt`, `images`, `image_sha256`, `id`, `condition`, `options`, `input_sha256`, `original_baseline_input_sha256`, `target_name`), exact rule block, fixed demos (`id`, `message.content`, `assistant_letter`), and pilot IDs. Demo labels/eligibility must be validated before construction; no new examples or rules are inferred here.
- `protocol.seal` computes stimulus and request identity. `checked_image` checks root containment and image hash. `verify_freeze(root)` needs caller-owned `freeze.json` containing a `hashes` mapping. Never use an untrusted freeze manifest.
- `api_backend.make_body(root, record)` verifies the sealed request and serializes original role/content order; `classify_receipt(receipt, record)` classifies a local receipt with `body_sha256`, `attempt_id`, `http_status`, `raw_body_base64` and `raw_body_sha256` (or `transport_error`). It does not call a provider or repair model output.
- `gpu_backend.encode_request(processor, request, root)` preserves original chat-template behavior and pinned `BASELINE_PARAMETERS`; model ID/revision constants remain. Model loading, GPU allocation and generation execution are excluded. A separate executor must honor those constants; this package does not certify GPU numerical equivalence.
- `run.Ledger` persists first-attempt events and rejects unresolved resends or altered receipts; `combine` uses the frozen parser. No launcher remains.
- `analyze.summarize(root)` needs `freeze.json`, `scoring-manifest.jsonl`, `final-scoring-manifest.jsonl`, `requests.jsonl`, optional `baseline-imported.jsonl`, and caller-produced `runtime/{gpu,api}` ledgers/responses. It writes `analysis/summary.json` beneath the explicit root. Missing outputs remain incomplete. The original 200 execution cohort and final common189 primary set must remain distinct.

Reuse requires identical original answer-only first attempts, inputs, options and model settings. Diagnostic answer-plus-evidence responses are not substitutes; changed target names require new matched inputs. These private provenance checks are the caller's responsibility because preparation/discovery was excluded. No prior spending cap is inherited.

Excluded: private preparation/discovery, fixed demos, orchestration, server/GPU identity checks, key/account handling, live API transport and model loading. Existing GPU-encoding and ledger tests remain with import-only changes. `code-provenance-fragment.json` records source-relative paths, original/output hashes and explicit modifications. Generated documentation is marked as new. The fragment intentionally excludes its own recursive hash.
