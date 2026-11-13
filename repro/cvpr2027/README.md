# EgoOwn evaluation code and portable analysis components

This release contains the frozen evaluation implementation used for the experiments and selected portable components from later studies. It is **code only**, not a complete replication bundle. Licensed images, item-level manifests/IDs, human review records, model outputs, private storage locations, credentials, model weights, and deployment infrastructure are not included.

## What is preserved

The ten `egoown_cvpr/*.py` modules are byte-identical to the frozen experiment package. The original package hash, computed as SHA-256 of sorted Python file names concatenated with their bytes, is:

`b6b605179323976fc371dfdb9393ec87cd054eb5670ee7d9f0a98dd093b7a0c7`

They provide target rendering, input construction, option ordering, the original answer parser and metrics, model interfaces, preflight, evaluation, and recovery helpers. `requirements-cvpr.txt` preserves the experiment dependency specification. The small integration in `vlmeval/dataset/egoownership.py` selects this input builder when explicitly configured. `scripts/run_cvpr_job.sh` is an environment-configured launcher and contains no credentials or deployment address.

The `api/`, `label_adaptation/`, and `controls/` directories contain extracted functions plus thin portable wrappers. They are **not byte-identical copies of the original complete suite runners**. Their READMEs identify what is preserved, changed, excluded, and required from the caller. `CODE_PROVENANCE.json` and the per-directory provenance records distinguish unchanged source from extracted/adapted code. The API transport is disabled by default; tests perform no model or network calls.

## Experimental contracts

- Standard inputs designate a target by a fixed red box and its name. Single uses the final frame; sparse uses three chronological frames. Source narrations, reference labels, taxonomy and review rationales are not model inputs.
- Open baseline settings use greedy generation, seed 0, a 256-token output limit, BF16, and disabled native thinking where supported. `configs/open-models.json` pins the historical revisions. Processor/runtime identity must also be checked for a rerun; this source release alone does not establish numerical runtime equivalence.
- Four prediction labels are MINE, OTHERS, SHARED and AMBIGUOUS. The latter represents insufficient attribution evidence. API refusal, error, truncation without a determinable answer, and parser rejection are not AMBIGUOUS answers and stay wrong in the denominator.
- Historical execution cohorts and final scoring cohorts differ. The initial open set contained 5,716 items; later open/small-API inputs contained 5,454. Final scoring uses 5,189 and the retained 189-item panel from an original 200. Configuration counts and manifest hashes are in `configs/protocol-reference.json`, without sample-level data.
- Baseline prompts precede 196 later target-name revisions in the retained full set, including four on the panel. Identical IDs alone do not justify input reuse. A changed target name requires a newly matched input; changed scoring labels alone may permit rescoring unchanged responses. Reuse must preserve original first-attempt bytes, prompts, image hashes/order, options, model/provider revisions and generation settings.
- The baseline scoring path uses the frozen original parser on directly executed raw responses. Separately contributed format-only S4 results belong to historical sensitivity analyses, not the default baseline. The S4 implementation is not included, and this release does not claim to reproduce it. Do not combine results with different extraction rules or response provenance as a controlled comparison.
- B0 removes the historical definitions/rules block; B1 keeps it; B2 adds the fixed four single-image demonstrations. Demonstrations and their item identities are not distributed. Restricted-view controls retain the full-window references. The effects do not establish hidden reasoning mechanisms or isolate every source of variation.

## Installation and offline checks

The existing repository installation instructions apply. For the historical evaluation environment, use `requirements-cvpr.txt` in a separate environment; exact GPU, processor and model availability need independent checks. The following synthetic tests require Python and Pillow, not GPU weights, licensed data or API credentials:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover -s tests -p test_cvpr_inputs.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest repro.cvpr2027.api.test_api_runner
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest repro.cvpr2027.label_adaptation.test_gpu_backend repro.cvpr2027.label_adaptation.test_runtime
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover -s repro/cvpr2027/controls -p 'test_*.py'
```

These targeted checks test input integrity, output parsing/denominators, paired analysis, payload/receipt validation, GPU encoding using fake processors, and first-attempt bookkeeping. Do not run unrelated repository API test suites expecting these commands to authorize paid calls.

## Data, execution and limitations

Obtain source data only through each original dataset's permitted access process. This repository does not grant redistribution or access rights. The caller must supply licensed source assets and the matching manifests, validate rendered PNGs and the input/reference version, and preserve a separate immutable record of every result. No personal review logs or generated examples are bundled as substitutes for the benchmark.

The public parameter templates disable execution and approve no spending. Previous experiment caps do not authorize a new run. Provider/model availability and prices must be checked separately before requesting a new budget. Deployment, cache management, credentials, account/key management, backup orchestration, historical recovery approvals, and private-data-dependent verification scripts are intentionally omitted. Therefore the release supports inspection and reuse of evaluation logic, but does not claim turnkey or complete numerical reproduction of every paper table.
