# EgoOwnership Benchmark — Usage Guide

A 4-way ownership classification task over egocentric video clips. Wraps the
HuggingFace dataset
[`Albertmade/ego-implicit-ownership-multiperson`](https://huggingface.co/datasets/Albertmade/ego-implicit-ownership-multiperson)
(3 parquet configs) into a VLMEvalKit MCQ task.

> **Note.** The dataset is *click-through gated* on HuggingFace. Anyone running
> this benchmark must have already accepted the gating prompt on the dataset
> page with the same account whose `HF_TOKEN` they pass in.

## What the task is

Given **up to 3 sparse frames** (t-2, t-1, t) of an egocentric clip plus a
narration sentence and verb/noun metadata, the model must decide who owns the
salient object the actor is interacting with:

| Letter | Label | Meaning |
|---|---|---|
| **A** | `MINE` | Owned by the camera wearer |
| **B** | `PERSON_k` | Owned by another visible person |
| **C** | `SHARED` | Communal / table-center, not personally owned |
| **D** | `AMBIGUOUS` | Symmetric / occluded / insufficient evidence |

There is **no human ground truth**. The reference label is the
`claude-jupiter-v1-p` judgement that ships in the parquet (`vlm_label`
column), so the accuracy a model achieves here measures **agreement with
Claude**, not absolute correctness. If you only want to score against the
heuristic rule label instead, set `EGOOWN_REF_FIELD=rule_label` (only the
`egolife` config populates that field for every row).

## Configs (dataset names)

| Dataset name | Source | N | Frames | Notes |
|---|---|---:|---|---|
| `EgoOwn` | HD-EPIC + EPIC-KITCHENS-100 (multi-person filter) | 564 | yes | Taxonomy C only |
| `EgoOwn_NarrA` | Ego4D FHO observer narrations | 3389 | **no** (text-only) | Taxonomy A only |
| `EgoOwn_EgoLife` | EgoLife | 899 | yes (888/899) | Taxonomies A/B/C/D |

The text-only `EgoOwn_NarrA` config sends no images to the model; the prompt
includes only the narration and clip metadata. Models without vision still run
on this config.

## Environment

The dataset class needs at minimum a HuggingFace token that has accepted the
gating prompt. Models also need their own API keys. Drop them into
`VLMEvalKit-EGO/.env`:

```sh
# .env
export HF_TOKEN=hf_...                     # must have accepted dataset gating
export ANTHROPIC_API_KEY=sk-ant-api03-...  # for Claude_Jupiter_V1_P, Claude4_*, etc.
export ANTHROPIC_BACKEND=official          # force the official Anthropic endpoint
# (optional dev) skip HF downloads — point at a local clone of the dataset repo
# export EGOOWN_LOCAL_ROOT=/path/to/ego-label-pipeline/hf_release
```

`run.py` auto-loads this file via the existing `load_env()` hook.

### Optional environment knobs

| Var | Default | Effect |
|---|---|---|
| `EGOOWN_REF_FIELD` | `vlm_label` | Reference label column. Set to `rule_label` to score against the heuristic. |
| `EGOOWN_LIMIT` | unset | If set to a positive int, only that many rows are evaluated. Useful for smoke tests — applied **before** frame downloads, so a 5-row run pulls only 15 jpgs. |
| `EGOOWN_LOCAL_ROOT` | unset | If set to a directory with `data/<file>.parquet` and `frames/...`, the class reads from there instead of HF. |

## Running an experiment

Single dataset + single model:

```sh
python run.py \
  --data EgoOwn \
  --model Claude_Jupiter_V1_P \
  --work-dir outputs/egoown \
  --api-nproc 8
```

All three configs at once, two API models in sequence:

```sh
python run.py \
  --data EgoOwn EgoOwn_NarrA EgoOwn_EgoLife \
  --model Claude_Jupiter_V1_P GPT4o_20241120 \
  --work-dir outputs/egoown
```

Local model via `--base-url` (LMDeploy / vLLM / SGLang / any OpenAI-compatible
server):

```sh
python run.py \
  --data EgoOwn \
  --base-url http://localhost:30000/v1 \
  --model Qwen2.5-VL-7B-Instruct \
  --work-dir outputs/egoown
```

Quick smoke run (5 rows):

```sh
EGOOWN_LIMIT=5 python run.py --data EgoOwn --model Claude_Jupiter_V1_P \
  --work-dir outputs/smoke_egoown
```

A minimal end-to-end smoke test that doesn't go through `run.py` lives in
`tests/smoke_egoown.py` — useful when iterating on the prompt or the
`evaluate()` function:

```sh
python tests/smoke_egoown.py
# or override:
EGOOWN_SMOKE_DATASET=EgoOwn_EgoLife EGOOWN_SMOKE_N=10 \
  EGOOWN_SMOKE_MODEL=Claude4_Sonnet python tests/smoke_egoown.py
```

## Output artefacts

Per `(model, dataset)` combination, under `outputs/<run-dir>/<model>/<eval_id>/`:

- `<model>_<dataset>.xlsx` — raw predictions (one row per clip)
- `<model>_<dataset>_scored.xlsx` — predictions + `pred_letter`, `pred_label`, `gt_label`, `hit`, `parsed`
- `<model>_<dataset>_acc.csv` — confusion matrix (rows = gt, cols = pred, with an extra `UNPARSED` column for predictions the regex couldn't reduce to A/B/C/D)
- `<model>_<dataset>_score.json` — scalar metrics:
  - `overall_acc`, `parsed_rate`, `n`
  - `acc:MINE`, `acc:PERSON_k`, `acc:SHARED`, `acc:AMBIGUOUS` (+ `n:<label>`)
  - `acc:taxonomy=A/B/C/D` (+ `n:taxonomy=…`)
  - `acc:source_dataset=hd_epic/epic_kitchens/ego4d_fho/egolife` (+ counts)

`run.py` also prints a `Run Summary Report` table at the end with
`overall_acc` as the primary metric.

## Adding a new model

Register a model in `vlmeval/config.py` (look for the `# Claude` section as a
template), then reference it by name with `--model`. Example — adding another
Anthropic model that goes through the official endpoint:

```python
"Claude_Jupiter_V1_P": partial(
    api.Claude3V,
    model="claude-jupiter-v1-p",
    backend="official",      # forces api.anthropic.com (no proxy)
    temperature=0,
    retry=6,
    max_tokens=1024,
    timeout=300,
),
```

## Architecture pointers

- Dataset class: [`vlmeval/dataset/egoownership.py`](../../vlmeval/dataset/egoownership.py)
  (`EgoOwnershipBench`)
- Registered in: [`vlmeval/dataset/__init__.py`](../../vlmeval/dataset/__init__.py)
- Model entry: [`vlmeval/config.py`](../../vlmeval/config.py) → `Claude_Jupiter_V1_P`
- Smoke test: [`tests/smoke_egoown.py`](../../tests/smoke_egoown.py)

The dataset class downloads parquet + frames lazily via `hf_hub_download` —
nothing is downloaded until `build_dataset(...)` is called, and frames are
fetched one-at-a-time the first time their relative path is referenced. The
HF hub cache (`~/.cache/huggingface/hub/...`) takes care of dedup across runs.

## Known limitations

- The reference labels are Claude's judgements, not human truth. Treat
  reported accuracies as **inter-rater agreement with Claude**.
- Sample sizes are small (564 / 3389 / 899); per-label accuracy in the rare
  classes (`PERSON_k` is 1/564 in `EgoOwn`) is noisy.
- The `EgoOwn_NarrA` config has no images — vision-only models without a text
  fallback will skip it.
- Some `EgoOwn_EgoLife` clips (11/899) had a frame extraction failure
  upstream and are evaluated with whatever frames did extract (0–2).
