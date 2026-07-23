#!/usr/bin/env python3
"""Cross-model error analysis for EgoOwn (§5.2 확장 ② + ③).

Inputs: per-item predictions from (a) VLM scored xlsx files and (b) probe
preds jsonl files. Computes:

  1. Pairwise error correlation (phi over wrong-indicators) + mean --
     "do models fail on the SAME items?" High correlation = structural,
     architecture-independent blind spot.
  2. Oracle best-of-k accuracy: items solved by >=1 model. If even the
     UNION of all models sits below the probe, the reasoning gap is not
     recoverable by ensembling.
  3. The all-wrong set: items no VLM solves; its label/taxonomy profile,
     and how many of them the probe solves ("reasoning gap set").
  4. Shortcut attribution (③): among each model's ERRORS on items where the
     proximity-cascade label (rule_label) disagrees with human GT, the
     fraction that equals rule_label -- agreement with the codified
     proximity heuristic. Reported overall and on taxonomy B (T2).

Usage:
  python scripts/egoown_error_analysis.py \
      --scored "outputs/**/*EgoOwn_scored.xlsx" \
      --probe-preds outputs/clip-probe/EgoOwn_probe_linear_groupcv_preds.jsonl \
      --out outputs/egoown_error_analysis.json
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
from glob import glob
from itertools import combinations

import numpy as np
import pandas as pd

_LABELS = ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS"]


def load_vlm_preds(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    need = {"index", "pred_label", "gt_label"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{path}: missing columns {need - set(df.columns)}")
    cols = ["index", "pred_label", "gt_label"] + [
        c for c in ("taxonomy", "rule_label", "source_dataset") if c in df.columns
    ]
    out = df[cols].copy()
    out["index"] = out["index"].astype(str)
    return out


def load_probe_preds(path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows).rename(columns={"pred_label": "pred_label", "gt_label": "gt_label"})
    df["index"] = df["index"].astype(str)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored", nargs="+", required=True,
                    help="scored xlsx paths or globs (sparse setting 권장)")
    ap.add_argument("--probe-preds", default=None,
                    help="probe *_preds.jsonl (reasoning-gap 교차용)")
    ap.add_argument("--out", default="egoown_error_analysis.json")
    args = ap.parse_args()

    paths = []
    for p in args.scored:
        paths.extend(sorted(glob(p, recursive=True)) if any(ch in p for ch in "*?") else [p])
    if len(paths) < 2:
        raise SystemExit(f"need >=2 scored files, got {paths}")

    models, meta = {}, None
    for p in paths:
        name = osp.basename(osp.dirname(p)) or osp.basename(p)
        df = load_vlm_preds(p)
        models[name] = df.set_index("index")
        if meta is None or len(df.columns) > len(meta.columns):
            meta = df.set_index("index")
    common = set.intersection(*[set(d.index) for d in models.values()])
    common = sorted(common)
    print(f"{len(models)} models, {len(common)} common items")

    wrong = {}   # model -> np.array bool (wrong)
    preds = {}
    for name, d in models.items():
        d = d.loc[common]
        wrong[name] = (d["pred_label"] != d["gt_label"]).to_numpy()
        preds[name] = d["pred_label"].to_numpy()
    gt = meta.loc[common, "gt_label"].to_numpy()

    # 1. pairwise error correlation (phi)
    def phi(a, b):
        a = a.astype(float); b = b.astype(float)
        va, vb = a.std(), b.std()
        return float(((a - a.mean()) * (b - b.mean())).mean() / (va * vb)) if va and vb else float("nan")
    pair = {f"{m1}|{m2}": round(phi(wrong[m1], wrong[m2]), 3)
            for m1, m2 in combinations(sorted(wrong), 2)}
    mean_phi = round(float(np.nanmean(list(pair.values()))), 3)

    # 2. oracle best-of-k
    any_right = ~np.logical_and.reduce(list(wrong.values()))
    oracle = round(float(any_right.mean()), 4)

    # 3. all-wrong set profile (+ probe rescue)
    all_wrong_mask = np.logical_and.reduce(list(wrong.values()))
    aw_idx = [common[i] for i in np.flatnonzero(all_wrong_mask)]
    profile = {"n": len(aw_idx)}
    for col in ("gt_label", "taxonomy"):
        if col == "gt_label" or (meta is not None and col in meta.columns):
            vals = gt[all_wrong_mask] if col == "gt_label" else meta.loc[aw_idx, col].to_numpy()
            profile[col] = pd.Series(vals).value_counts().to_dict()
    probe_stats = None
    if args.probe_preds:
        pr = load_probe_preds(args.probe_preds).set_index("index")
        inter = [i for i in aw_idx if i in pr.index]
        solved = sum(pr.loc[i, "pred_label"] == meta.loc[i, "gt_label"] for i in inter)
        probe_stats = {"all_wrong_n": len(aw_idx), "probe_covers": len(inter),
                       "probe_solves": int(solved),
                       "probe_solve_rate_on_all_wrong": round(solved / max(1, len(inter)), 4)}

    # 4. shortcut attribution vs rule_label (③)
    shortcut = {}
    if meta is not None and "rule_label" in meta.columns:
        rl = meta.loc[common, "rule_label"].to_numpy()
        conflict = (rl != gt) & pd.Series(rl).isin(_LABELS).to_numpy()
        tax = meta.loc[common, "taxonomy"].to_numpy() if "taxonomy" in meta.columns else None
        for name in sorted(preds):
            e = wrong[name] & conflict
            n_err = int(e.sum())
            agree = int(((preds[name] == rl) & e).sum())
            row = {"n_conflict_errors": n_err,
                   "follows_proximity_rate": round(agree / max(1, n_err), 4)}
            if tax is not None:
                eb = e & (tax == "B")
                row["T2_n"] = int(eb.sum())
                row["T2_follows_proximity_rate"] = round(
                    int(((preds[name] == rl) & eb).sum()) / max(1, int(eb.sum())), 4)
            shortcut[name] = row

    result = {"n_models": len(models), "n_common_items": len(common),
              "mean_pairwise_error_phi": mean_phi, "pairwise_error_phi": pair,
              "oracle_best_of_k_acc": oracle,
              "all_wrong_profile": profile,
              "probe_on_all_wrong": probe_stats,
              "shortcut_attribution_vs_rule_label": shortcut}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False, default=str)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("pairwise_error_phi",)}, indent=2, default=str)[:2000])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
