#!/usr/bin/env python3
"""Paper figures for EgoOwn (AAAI-27).

Generates, from eval artifacts produced by `egoown_report.py` and the
per-run `*_acc.csv` confusion matrices:

  F1  fig1_teaser.pdf            — motivation bar chart: overall acc per model
                                   (sparse mode) vs. Human / linear-probe /
                                   chance reference lines.
  F4  fig4_confusion_collapse.pdf — 2-panel: (a) row-normalized confusion
                                   matrices for 2–3 representative models,
                                   (b) prediction-distribution "label collapse"
                                   chart (pred_frac stacked bars vs. GT prior).

Usage (run from repo root, after `python scripts/egoown_report.py --outputs ./outputs`):

  # F1 teaser
  python scripts/egoown_figures.py f1 \
      --report egoown_report.csv --dataset EgoOwn --human 0.834 \
      --out figures/fig1_teaser

  # F4 confusion + collapse (model dirs as they appear under --outputs)
  python scripts/egoown_figures.py f4 \
      --outputs ./outputs --report egoown_report.csv --dataset EgoOwn \
      --models "Qwen2.5-VL-32B-Instruct:Qwen2.5-VL-32B" \
               "EgoThinker:EgoThinker" \
               "gpt-5.4-mini-2026-03-17:GPT-5.4-mini" \
      --out figures/fig4_confusion_collapse

Notes
-----
* `--models` entries are `<model_dir>[:<display name>]`; `<model_dir>` must
  match the directory name under --outputs.
* Only seed-0 runs are used (paths containing `optseed` are skipped, and
  report rows with opt_seed not in {"", 0} are dropped).
* Outputs both .pdf (for LaTeX) and .png (preview) for each figure.
* Colors are Okabe–Ito (colorblind-safe); fonts embedded as TrueType
  (pdf.fonttype=42) per AAAI camera-ready requirements.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import re
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants

_LABELS = ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS"]
# Display names must match the paper macros (\lmine, \lothers, \lshared, \lambig).
_DISPLAY = {"MINE": "Mine", "PERSON_k": "Others'", "SHARED": "Shared",
            "AMBIGUOUS": "Ambiguous", "UNPARSED": "Unparsed"}

# Label colors: Okabe–Ito (colorblind-safe 4-way distinction — do not swap
# for brand colors). Accent colors: Meta palette (#0668E1/#0080FB/#1C2B33).
_C = {"MINE": "#0072B2", "PERSON_k": "#E69F00", "SHARED": "#009E73",
      "AMBIGUOUS": "#CC79A7", "UNPARSED": "#999999",
      "bar": "#0668E1", "probe": "#0080FB", "human": "#D55E00",
      "chance": "#1C2B33"}

_PROBE_PAT = re.compile(r"clip|siglip|egovlp|probe", re.IGNORECASE)

# Non-canonical artifacts swept up by egoown_report.py's recursive glob:
# smoke/exploratory output trees, and nested per-run T<ts> workdir copies whose
# basename becomes a bogus "model" name. These duplicate or predate the canonical
# top-level v2 runs and must not become figure bars.
_JUNK_TREE = re.compile(r"(?:^|/)(?:smoke|smoke_hf|vllm_smoke|vllm_full)/")
_TWORKDIR = re.compile(r"(?:^|/)T\d{8}-\d{6}/")
_TNAME = re.compile(r"^T\d{8}-\d{6}$")
# For probes, the leakage-free grouped-CV linear probe is the paper's headline.
_PROBE_METRIC = "probe-linear_groupcv"

def _register_font(name: str = "Aptos") -> None:
    """Register TTF/OTF files from scripts/fonts/ (if present) and select
    `name` with silent fallback to the matplotlib default."""
    import matplotlib.font_manager as fm
    fdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    if os.path.isdir(fdir):
        for f in glob(os.path.join(fdir, "*.[ot]tf")):
            fm.fontManager.addfont(f)
    available = {f.name for f in fm.fontManager.ttflist}
    if name in available:
        plt.rcParams["font.family"] = name
    else:
        print(f"[warn] font '{name}' not found — using default "
              f"(drop TTFs into {fdir} to enable)")


plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
})
_register_font("Aptos")


# ------------------------------------------------------------------ helpers

def _load_report(path: str, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["dataset"] == dataset].copy()
    # drop non-canonical rows: smoke/exploratory trees + nested T<ts> run-workdir
    # copies + their basename-derived pseudo-model rows.
    sf = df["score_file"].fillna("").astype(str)
    df = df[~sf.str.contains(_JUNK_TREE) & ~sf.str.contains(_TWORKDIR)]
    df = df[~df["model"].astype(str).str.match(_TNAME)]
    # honor the paper's explicit exclusions (e.g. the 72B OpenRouter serving artifact)
    if "exclude_from_main_table" in df.columns:
        df = df[df["exclude_from_main_table"] != True]  # noqa: E712
    # probes emit several metrics (zeroshot / item-CV / grouped-CV); keep only the
    # leakage-free grouped-CV linear probe so each backbone is a single bar.
    is_probe = df["model"].astype(str).str.contains(_PROBE_PAT)
    if "mode" in df.columns:
        df = df[~is_probe | (df["mode"].astype(str) == _PROBE_METRIC)]
    # seed-0 runs only (permutation sweeps live in optseed workdirs / rows)
    if "opt_seed" in df.columns:
        seed = df["opt_seed"].fillna("").astype(str)
        df = df[seed.isin(["", "0", "0.0"])]
    if df.empty:
        raise SystemExit(f"No rows for dataset={dataset} (seed 0) in {path}")
    # one row per model — keep the latest run
    if "timestamp_utc" in df.columns:
        df = df.sort_values("timestamp_utc").groupby("model", as_index=False).last()
    return df


def _parse_models(specs: list[str]) -> list[tuple[str, str]]:
    out = []
    for s in specs:
        mdir, _, disp = s.partition(":")
        out.append((mdir, disp or mdir))
    return out


def _find_acc_csv(outputs: str, model_dir: str, dataset: str) -> str:
    """Locate the seed-0 confusion CSV for (model, dataset).

    Matches `*_{dataset}_acc.csv` exactly — `_EgoOwn_acc.csv` does NOT match
    `_EgoOwn_Single_acc.csv`, so mode suffixes stay separated.
    """
    hits = [
        p for p in glob(osp.join(outputs, model_dir, "**", f"*_{dataset}_acc.csv"),
                        recursive=True)
        if "optseed" not in p
    ]
    if not hits:
        raise SystemExit(
            f"No *_{dataset}_acc.csv under {osp.join(outputs, model_dir)} "
            f"(seed-0). Check the model dir name (`ls {outputs}`).")
    if len(hits) > 1:
        hits.sort(key=os.path.getmtime)
        print(f"[warn] {len(hits)} acc.csv candidates for {model_dir}; "
              f"using newest: {hits[-1]}")
    return hits[-1]


def _load_confusion(path: str) -> pd.DataFrame:
    # acc.csv is a headerless-index square matrix: the header row names the
    # predicted labels, and the data rows are the GT classes in the SAME order
    # (no row-label column). Assign the GT index from the column labels.
    cm = pd.read_csv(path)
    cm.index = pd.Index([str(c) for c in cm.columns][: len(cm)])
    # GT is never UNPARSED — drop the all-zero row, keep the pred column.
    cm = cm.loc[[l for l in _LABELS if l in cm.index]]
    cols = [c for c in _LABELS + ["UNPARSED"] if c in cm.columns]
    return cm[cols]


def _save(fig, out_prefix: str):
    os.makedirs(osp.dirname(out_prefix) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=300, bbox_inches="tight")
    print(f"[ok] wrote {out_prefix}.pdf / .png")


# ----------------------------------------------------------------------- F1

def make_f1(args):
    df = _load_report(args.report, args.dataset)
    df = df.dropna(subset=["overall_acc"]).sort_values("overall_acc")
    names = [_clean_name(m, args.rename) for m in df["model"]]
    is_probe = df["model"].str.contains(_PROBE_PAT)
    colors = [_C["probe"] if p else _C["bar"] for p in is_probe]

    fig, ax = plt.subplots(figsize=(3.3, 0.28 * len(df) + 0.9))
    ax.barh(names, df["overall_acc"], color=colors, height=0.62)
    ax.axvline(args.chance, color=_C["chance"], ls=":", lw=1)
    ax.axvline(args.human, color=_C["human"], ls="--", lw=1.2)
    from matplotlib.transforms import blended_transform_factory
    tf = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(args.chance, 1.01, "chance", color=_C["chance"], fontsize=6.5,
            ha="center", va="bottom", transform=tf)
    ax.text(args.human, 1.01, f"human {args.human:.2f}", color=_C["human"],
            fontsize=6.5, ha="center", va="bottom", transform=tf)
    for y, v in enumerate(df["overall_acc"]):
        ax.text(v + 0.008, y, f"{v:.2f}", va="center", fontsize=6.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Overall accuracy (sparse, n=3,227)")
    if is_probe.any():
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=_C["bar"], label="VLM"),
                           Patch(color=_C["probe"], label="Frozen-feature probe")],
                  loc="lower right", frameon=False)
    _save(fig, args.out)


def _clean_name(m: str, renames: list[str]) -> str:
    for r in renames or []:
        old, _, new = r.partition("=")
        if old == m:
            return new
    m = re.sub(r"-Instruct$|-\d{4}-\d{2}-\d{2}$", "", m)
    return m


# ----------------------------------------------------------------------- F4

def make_f4(args):
    models = _parse_models(args.models)
    n = len(models)

    fig = plt.figure(figsize=(7.6, 2.1))
    # trailing spacer column keeps panel (b)'s long y-labels off the last heatmap
    gs = fig.add_gridspec(1, n + 2, width_ratios=[1.0] * n + [0.55, 1.35],
                          wspace=0.32)

    # -- (a) confusion heatmaps, row-normalized ---------------------------
    tick_disp = [_DISPLAY[l] for l in _LABELS]
    for i, (mdir, disp) in enumerate(models):
        cm = _load_confusion(_find_acc_csv(args.outputs, mdir, args.dataset))
        norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for r in range(norm.shape[0]):
            for c in range(norm.shape[1]):
                v = norm.iloc[r, c]
                if v >= 0.005:
                    ax.text(c, r, f"{v:.2f}".lstrip("0"), ha="center",
                            va="center", fontsize=5.8,
                            color="white" if v > 0.55 else "black")
        ax.set_xticks(range(norm.shape[1]))
        ax.set_xticklabels([_DISPLAY.get(c, c) for c in norm.columns],
                           rotation=45, ha="right")
        ax.set_yticks(range(len(tick_disp)))
        ax.set_yticklabels(tick_disp if i == 0 else [])
        if i == 0:
            ax.set_ylabel("Ground truth")
        ax.set_title(disp, pad=3)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text(0.02, 0.97, "(a)", fontsize=9, fontweight="bold")

    # -- (b) label-collapse: pred_frac stacked bars ------------------------
    ax = fig.add_subplot(gs[0, n + 1])
    df = _load_report(args.report, args.dataset)
    df = df[~df["model"].str.contains(_PROBE_PAT)]
    pf_cols = [f"pred_frac:{l}" for l in _LABELS]
    df = df.dropna(subset=pf_cols)
    # sort by max single-label mass = collapse severity
    df["_collapse"] = df[pf_cols].max(axis=1)
    df = df.sort_values("_collapse")
    names = [_clean_name(m, args.rename) for m in df["model"]]
    mat = df[pf_cols].values
    unparsed = np.clip(1.0 - mat.sum(axis=1), 0, 1)

    rows = names + ["GT prior"]
    gt = np.array(args.gt_prior)
    mat = np.vstack([mat, gt])
    unparsed = np.append(unparsed, 0.0)

    left = np.zeros(len(rows))
    for j, l in enumerate(_LABELS):
        ax.barh(rows, mat[:, j], left=left, color=_C[l], height=0.68,
                label=_DISPLAY[l])
        left += mat[:, j]
    ax.barh(rows, unparsed, left=left, color=_C["UNPARSED"], height=0.68,
            label="Unparsed")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Fraction of predictions")
    ax.axhline(len(rows) - 1.5, color="black", lw=0.6, ls="-")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.38),
              frameon=False, handlelength=1.0, columnspacing=0.8)
    ax.set_title("Prediction distribution", pad=3)
    fig.text(0.60, 0.97, "(b)", fontsize=9, fontweight="bold")

    _save(fig, args.out)


# ----------------------------------------------------------------------- F5

def make_sampeff(args):
    """Probe sample-efficiency curve: grouped-CV accuracy vs #train labels,
    with the best zero-shot VLM / human / chance reference lines. Shows that a
    few dozen labels already clear every VLM (the 'supervision advantage'
    rebuttal)."""
    import json
    with open(args.curve) as fh:
        curve = json.load(fh)["curve"]
    curve = sorted(curve, key=lambda r: r["approx_train_labels"])
    x = np.array([r["approx_train_labels"] for r in curve], dtype=float)
    y = np.array([r["acc_mean"] for r in curve])
    e = np.array([r.get("acc_std", 0.0) for r in curve])

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.set_xscale("log")
    ax.fill_between(x, y - e, y + e, color=_C["bar"], alpha=0.15, linewidth=0)
    ax.plot(x, y, "-o", color=_C["bar"], lw=2, ms=6, zorder=5)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=6.5, color=_C["bar"])

    from matplotlib.transforms import blended_transform_factory
    tf = blended_transform_factory(ax.transAxes, ax.transData)
    # reference lines; labels split left/right so none sits under the curve
    for val, col, lab, xpos, ha in [
        (args.human, _C["human"], f"human {args.human:.2f}", 0.015, "left"),
        (args.best_vlm, _C["PERSON_k"],
         f"{args.best_vlm_label} {args.best_vlm:.2f}", 0.985, "right"),
        (args.chance, _C["chance"], f"chance {args.chance:.2f}", 0.015, "left"),
    ]:
        ax.axhline(val, color=col, ls="--", lw=1.1, zorder=1)
        ax.text(xpos, val + 0.013, lab, color=col, fontsize=6.5,
                ha=ha, va="bottom", transform=tf)

    # crossover punchline: smallest budget already clears the best VLM
    if y[0] > args.best_vlm:
        ax.annotate(f"≈{int(x[0])} labels\nbeat best VLM", (x[0], y[0]),
                    textcoords="offset points", xytext=(20, 6), fontsize=6.5,
                    color=_C["bar"], va="center",
                    arrowprops=dict(arrowstyle="->", color=_C["bar"], lw=0.8))

    ax.set_xlabel("Labeled training examples (log scale)")
    ax.set_ylabel("Grouped-CV accuracy")
    ax.set_title("CLIP frozen-feature probe: sample efficiency", pad=6)
    ax.set_ylim(0.2, 0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in x])
    ax.tick_params(axis="x", which="minor", length=0)
    _save(fig, args.out)


# ---------------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f1 = sub.add_parser("f1", help="teaser bar chart")
    f1.add_argument("--report", default="egoown_report.csv")
    f1.add_argument("--dataset", default="EgoOwn",
                    help="report `dataset` value (EgoOwn = sparse)")
    f1.add_argument("--human", type=float, default=0.834)
    f1.add_argument("--chance", type=float, default=0.25)
    f1.add_argument("--rename", nargs="*", default=[],
                    help="model display renames: OLD=NEW ...")
    f1.add_argument("--out", default="figures/fig1_teaser")
    f1.set_defaults(fn=make_f1)

    f4 = sub.add_parser("f4", help="confusion + label-collapse 2-panel")
    f4.add_argument("--outputs", default="./outputs")
    f4.add_argument("--report", default="egoown_report.csv")
    f4.add_argument("--dataset", default="EgoOwn")
    f4.add_argument("--models", nargs="+", required=True,
                    help="2–3 entries: <model_dir>[:<display>]")
    f4.add_argument("--gt-prior", nargs=4, type=float,
                    default=[0.358, 0.379, 0.216, 0.046],
                    metavar=("MINE", "PERSON_k", "SHARED", "AMBIG"),
                    help="benchmark label prior (n=3,227 defaults)")
    f4.add_argument("--rename", nargs="*", default=[])
    f4.add_argument("--out", default="figures/fig4_confusion_collapse")
    f4.set_defaults(fn=make_f4)

    f5 = sub.add_parser("sampeff", help="probe sample-efficiency curve")
    f5.add_argument("--curve",
                    default="outputs/clip-probe/EgoOwn_probe_sampeff_curve.json")
    f5.add_argument("--best-vlm", type=float, default=0.544,
                    help="best zero-shot VLM sparse acc (reference line)")
    f5.add_argument("--best-vlm-label", default="best zero-shot VLM")
    f5.add_argument("--human", type=float, default=0.834)
    f5.add_argument("--chance", type=float, default=0.25)
    f5.add_argument("--out", default="figures/fig5_sample_efficiency")
    f5.set_defaults(fn=make_sampeff)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
