#!/usr/bin/env bash
# Unified EgoOwn sweep: models × input modes, with a reproducibility header.
#
# Usage:
#   MODELS="GPT4o Qwen2.5-VL-7B-Instruct" ./scripts/run_egoown_sweep.sh
#   MODELS="GPT4o" DATASETS="EgoOwn_Single EgoOwn" EGOOWN_LIMIT=50 ./scripts/run_egoown_sweep.sh   # smoke
#   MODELS="GPT4o" PERMUTE_SEEDS="0 1 2" ./scripts/run_egoown_sweep.sh   # §5.4 option-order permutation
#
# Notes:
# - EGOOWN_REF_FIELD: set to human_label once the human re-review lands.
# - EgoOwn_Clip needs EGOOWN_VIDEOS_ROOT (source {video_id}.mp4 files).
# - Every run's *_score.json embeds a manifest (prompt version, seed, git rev);
#   scripts/egoown_report.py refuses silence on mixed versions.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:?Set MODELS, e.g. MODELS=\"GPT4o Qwen2.5-VL-7B-Instruct\"}"
DATASETS="${DATASETS:-EgoOwn_Single EgoOwn EgoOwn_Blind}"
PERMUTE_SEEDS="${PERMUTE_SEEDS:-0}"
WORK_DIR="${WORK_DIR:-./outputs}"

echo "=== EgoOwn sweep ==="
echo "git rev:   $(git rev-parse --short HEAD) (dirty: $(git diff --quiet && echo no || echo YES))"
echo "models:    ${MODELS}"
echo "datasets:  ${DATASETS}"
echo "opt seeds: ${PERMUTE_SEEDS}"
echo "ref field: ${EGOOWN_REF_FIELD:-vlm_label (default)}"
echo "limit:     ${EGOOWN_LIMIT:-0 (all)}"
echo "===================="

for seed in ${PERMUTE_SEEDS}; do
  for model in ${MODELS}; do
    for dataset in ${DATASETS}; do
      echo ">>> model=${model} dataset=${dataset} opt_seed=${seed}"
      EGOOWN_OPT_SEED="${seed}" python run.py \
        --model "${model}" \
        --data "${dataset}" \
        --work-dir "${WORK_DIR}" \
        --reuse \
        || { echo "!!! FAILED: ${model} × ${dataset} (seed ${seed}) — continuing"; }
    done
  done
done

python scripts/egoown_report.py --outputs "${WORK_DIR}" --out-prefix "${WORK_DIR}/egoown"
echo "Report: ${WORK_DIR}/egoown_report.csv, ${WORK_DIR}/egoown_main_table.md"
