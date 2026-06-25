"""End-to-end smoke test for the EgoOwnership dataset.

Runs N samples through Claude (`claude-jupiter-v1-p`), constructs a
prediction DataFrame in the shape `evaluate()` expects, and reports
overall + per-label accuracy. Designed to finish in well under a minute
with N=3.

Run:
    cd /home/gpuadmin/albert/VLMEvalKit-EGO
    source /home/gpuadmin/albert/ego-label-pipeline/.env
    export EGOOWN_LOCAL_ROOT=/home/gpuadmin/albert/ego-label-pipeline/hf_release
    python tests/smoke_egoown.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

# Force the Claude wrapper to use the official Anthropic endpoint.
os.environ.setdefault("ANTHROPIC_BACKEND", "official")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vlmeval.config import supported_VLM  # noqa: E402
from vlmeval.dataset import build_dataset  # noqa: E402
from vlmeval.smp import dump  # noqa: E402

N = int(os.environ.get("EGOOWN_SMOKE_N", "3"))
DATASET_NAME = os.environ.get("EGOOWN_SMOKE_DATASET", "EgoOwn")
MODEL_NAME = os.environ.get("EGOOWN_SMOKE_MODEL", "Claude_Jupiter_V1_P")


def main() -> int:
    print(f"[smoke] dataset={DATASET_NAME} model={MODEL_NAME} N={N}")

    ds = build_dataset(DATASET_NAME)
    print(f"[smoke] built dataset rows={len(ds.data)} TYPE={ds.TYPE} MODALITY={ds.MODALITY}")

    rows = ds.data.head(N).copy().reset_index(drop=True)
    model = supported_VLM[MODEL_NAME]()
    print(f"[smoke] built model {MODEL_NAME}")

    predictions = []
    for i in range(len(rows)):
        line = rows.iloc[i]
        msgs = ds.build_prompt(line)
        t0 = time.time()
        try:
            pred = model.generate(msgs, dataset=DATASET_NAME)
        except Exception as exc:  # noqa: BLE001
            pred = f"[ERROR] {type(exc).__name__}: {exc}"
        dt = time.time() - t0
        print(
            f"[smoke] {i+1}/{len(rows)} clip={line['clip_id']} "
            f"gt_letter={line['answer']} pred={str(pred)[:80]!r} ({dt:.1f}s)"
        )
        predictions.append(pred)

    rows["prediction"] = predictions

    # Persist to a temp xlsx the way infer_data_job would, then evaluate().
    with tempfile.TemporaryDirectory(prefix="egoown_smoke_") as tmp:
        eval_file = Path(tmp) / f"{MODEL_NAME}_{DATASET_NAME}.xlsx"
        dump(rows, str(eval_file))
        print(f"[smoke] wrote predictions -> {eval_file}")
        results = ds.evaluate(str(eval_file))
        print("[smoke] evaluate() returned:")
        for k, v in results.items():
            print(f"    {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
