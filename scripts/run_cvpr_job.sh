#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_TOKEN_PATH:?Provide a private token file through the job secret mount}"
: "${DATASET_REPO:?}" "${DATASET_REVISION:?}" "${RESULTS_REPO:?}" "${MODEL:?}" "${RUN_DIR:?}" "${MODELS_JSON:?}"
export HF_TOKEN
HF_TOKEN="$(cat "$HF_TOKEN_PATH")"
export TOKENIZERS_PARALLELISM=false
python -m egoown_cvpr.hub download --repo "$DATASET_REPO" --revision "$DATASET_REVISION" --root "${DATA_ROOT:-/workspace/egoown/data}"
python -m egoown_cvpr.run --root "${DATA_ROOT:-/workspace/egoown/data}" --models "$MODELS_JSON" --model "$MODEL" --condition "${CONDITION:-sparse}" --subset "${SUBSET:-full}" --seed "${OPT_SEED:-0}" --out "$RUN_DIR" --results-repo "$RESULTS_REPO"
