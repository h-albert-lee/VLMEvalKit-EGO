#!/usr/bin/env python3
"""Aggregate EgoOwn *_score.json files into one consistent report.

Walks --outputs (VLMEvalKit's output dir layout: <outputs>/<model>/*_score.json),
collects every EgoOwn* score file, and emits:

  egoown_report.csv   — long-form: one row per (model, dataset/mode) with all
                        scalar metrics + manifest fields (prompt_version,
                        opt_seed, ref_field, eval_code_rev, timestamp)
  egoown_main_table.md — the paper's main-table view: models × modes with
                        per-label acc / macro-F1 / per-taxonomy acc /
                        abstention metrics

Usage:
    python scripts/egoown_report.py --outputs ./outputs
    python scripts/egoown_report.py --outputs ./outputs --out-prefix results/egoown
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from glob import glob

import pandas as pd

_MAIN_COLS = [
    "acc:MINE", "acc:PERSON_k", "acc:SHARED", "acc:AMBIGUOUS", "macro_f1",
    "acc:taxonomy=T1", "acc:taxonomy=T2", "acc:taxonomy=T3", "acc:taxonomy=T4",
    "abstain_precision", "abstain_recall", "over_abstention_rate",
    "overall_acc", "parsed_rate", "n",
]


def collect(outputs_dir: str) -> pd.DataFrame:
    rows = []
    pattern = osp.join(outputs_dir, "**", "*EgoOwn*_score.json")
    for path in sorted(glob(pattern, recursive=True)):
        try:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[skip] {path}: {exc}")
            continue
        manifest = blob.pop("manifest", {}) or {}
        model = osp.basename(osp.dirname(path))
        row = {
            "model": model,
            "dataset": manifest.get("dataset") or _dataset_from_filename(path),
            "mode": manifest.get("mode", ""),
            "prompt_version": manifest.get("prompt_version", ""),
            "opt_seed": manifest.get("opt_seed", ""),
            "ref_field": manifest.get("ref_field", ""),
            "eval_code_rev": manifest.get("eval_code_rev", ""),
            "timestamp_utc": manifest.get("timestamp_utc", ""),
            "score_file": osp.relpath(path, outputs_dir),
        }
        row.update({k: v for k, v in blob.items() if not isinstance(v, (dict, list))})
        rows.append(row)
    if not rows:
        raise SystemExit(f"No *EgoOwn*_score.json found under {outputs_dir}")
    return pd.DataFrame(rows)


def _dataset_from_filename(path: str) -> str:
    base = osp.basename(path)
    for tok in base.split("_"):
        if tok.startswith("EgoOwn"):
            return tok
    return "EgoOwn"


def main_table_md(df: pd.DataFrame) -> str:
    cols = [c for c in _MAIN_COLS if c in df.columns]
    view = df[["model", "dataset", "mode", "opt_seed"] + cols].copy()
    view = view.sort_values(["dataset", "mode", "model", "opt_seed"])
    for c in cols:
        if c not in ("n",):
            view[c] = view[c].map(
                lambda v: f"{v:.3f}" if isinstance(v, (int, float)) and pd.notna(v) else "—"
            )
    lines = [
        "# EgoOwn main results",
        "",
        "Grouped by dataset config + input mode. `—` = label absent / unmeasured.",
        "",
        view.to_markdown(index=False),
        "",
    ]
    # Reproducibility footer: flag any mixed prompt versions or code revs.
    for field in ("prompt_version", "ref_field", "eval_code_rev"):
        vals = sorted(set(df[field].astype(str)) - {""})
        marker = "" if len(vals) <= 1 else "  ⚠️ MIXED — results not comparable!"
        lines.append(f"- {field}: {', '.join(vals) or 'unknown'}{marker}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", default="./outputs", help="VLMEvalKit outputs dir")
    ap.add_argument("--out-prefix", default="egoown", help="Output file prefix")
    args = ap.parse_args()

    df = collect(args.outputs)
    os.makedirs(osp.dirname(args.out_prefix) or ".", exist_ok=True)

    csv_path = f"{args.out_prefix}_report.csv"
    md_path = f"{args.out_prefix}_main_table.md"
    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(main_table_md(df))
    print(f"Collected {len(df)} runs → {csv_path}, {md_path}")


if __name__ == "__main__":
    main()
