#!/usr/bin/env python3
"""CLIP/SigLIP representation probe for the EgoOwn benchmark (paper §5, Table 2).

Answers: *is ownership information present in frozen visual representations
at all?* Two probes over the same items the VLMs are evaluated on:

  zero-shot  cosine similarity between the image embedding and per-label text
             prompts (multiple templates averaged). CAVEAT: AMBIGUOUS has no
             visual prototype — its templates describe "unclear ownership",
             which is a weak proxy; interpret the AMBIGUOUS column loosely.
  linear     multinomial logistic regression on frozen features with
             stratified k-fold; metrics computed on out-of-fold predictions
             (every item scored exactly once by a model that never saw it).

Features per item: embedding of the target frame t, and — when the parquet
carries a target bbox — the bbox crop embedding concatenated (the probe then
knows *which* object is in question, mirroring the red-box shown to VLMs).

Outputs land in the VLMEvalKit outputs layout so `scripts/egoown_report.py`
picks them up automatically as extra model rows:

    <out-dir>/<probe-name>/<dataset>_probe_<zeroshot|linear>_score.json
    ..._acc.csv (confusion matrix)

Metrics schema mirrors EgoOwnershipBench.evaluate(): per-label acc/F1,
macro_f1, per-taxonomy acc, abstention P/R, over-abstention, pred_frac,
manifest (prompt/probe version, ref_field, git rev).

Usage (on the eval server):
    python scripts/egoown_probe.py --dataset EgoOwn --model clip
    python scripts/egoown_probe.py --dataset EgoOwn --model siglip \\
        --probe linear --folds 5
    EGOOWN_REF_FIELD=human_label EGOOWN_LIMIT=200 \\
        python scripts/egoown_probe.py --dataset EgoOwn --model clip   # smoke

Deps: torch, transformers, pillow, scikit-learn (linear probe only).
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))

PROBE_VERSION = "probe-v1-2026-07-13"
_LABELS = ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS"]
_ABSTAIN = "AMBIGUOUS"

_MODEL_ALIASES = {
    "clip": "openai/clip-vit-large-patch14",
    "siglip": "google/siglip-so400m-patch14-384",
}

# Zero-shot text templates per label; {noun} is substituted with the target
# object noun when known. Kept deliberately simple — the probe measures
# whether the representation supports these distinctions, not prompt tuning.
_ZS_TEMPLATES = {
    "MINE": [
        "my own {noun}, right in front of me",
        "a first-person photo of my {noun}",
        "the camera wearer's own {noun}",
    ],
    "PERSON_k": [
        "someone else's {noun}",
        "another person's {noun} on their side of the table",
        "a {noun} that belongs to the other person",
    ],
    "SHARED": [
        "a shared {noun} for everyone at the table",
        "a communal {noun} used by several people",
        "a {noun} placed in the middle for common use",
    ],
    "AMBIGUOUS": [
        "a {noun} whose owner is unclear",
        "an unattended {noun} that could belong to anyone",
        "a {noun} with no sign of who owns it",
    ],
}


def _git_rev(path: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_items(dataset: str):
    """Rows via EgoOwnershipBench (frames resolved, ref labels filtered),
    plus a clip_id -> (bbox, noun) map read directly from the parquet."""
    from vlmeval.dataset.egoownership import (  # noqa: PLC0415
        _BASE_CONFIGS, _parse_dataset_name, EgoOwnershipBench,
    )

    base, _, _ = _parse_dataset_name(dataset)  # (base, mode, prompt_style)
    bench = EgoOwnershipBench(base)  # sparse mode: image_path = [t-2, t-1, t]
    df = bench.data

    bbox_by_id: dict[str, dict] = {}
    noun_by_id: dict[str, str] = {}
    try:
        parquet = bench._download_parquet(_BASE_CONFIGS[base][0])
        raw = pd.read_parquet(parquet)
        if "clip_id" in raw.columns and "object" in raw.columns:
            for cid, obj in zip(raw["clip_id"].astype(str), raw["object"]):
                if isinstance(obj, dict):
                    if obj.get("bbox") is not None:
                        bbox_by_id[cid] = obj["bbox"]
                    if obj.get("label"):
                        noun_by_id[cid] = str(obj["label"])
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] no bbox/noun metadata ({exc}) — full-frame features only")

    items = []
    for _, row in df.iterrows():
        paths = row["image_path"]
        if isinstance(paths, np.ndarray):
            paths = list(paths)
        if not paths:
            continue
        cid = str(row["index"])
        items.append({
            "id": cid,
            "frame_t": paths[-1],  # last path is frame t in sparse layout
            "bbox": bbox_by_id.get(cid),
            "noun": noun_by_id.get(cid, "object"),
            "label": row["answer_label"],
            "taxonomy": row.get("taxonomy"),
            "source_dataset": row.get("source_dataset"),
            "video_id": row.get("video_id"),
        })
    print(f"[probe] {len(items)} items with frame t "
          f"({sum(1 for i in items if i['bbox'])} with bbox)")
    return items, bench


# ---------------------------------------------------------------------------
# Embedding backends (lazy torch import)
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(self, model_id: str, batch_size: int = 32):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model_id = model_id
        self.batch_size = batch_size

    def embed_images(self, images) -> np.ndarray:
        import torch
        feats = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self.model.get_image_features(**inputs)
            out = out / out.norm(dim=-1, keepdim=True)
            feats.append(out.float().cpu().numpy())
            if (i // self.batch_size) % 10 == 0:
                print(f"  embed {i + len(batch)}/{len(images)}", end="\r")
        print()
        return np.concatenate(feats, axis=0)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        import torch
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.inference_mode():
            out = self.model.get_text_features(**inputs)
        out = out / out.norm(dim=-1, keepdim=True)
        return out.float().cpu().numpy()


def load_images(items, *, crop: bool):
    """PIL images: full frame t, or bbox crops (fallback to full frame)."""
    from PIL import Image
    images = []
    for it in items:
        img = Image.open(it["frame_t"]).convert("RGB")
        if crop and it["bbox"]:
            b = it["bbox"]
            w, h = img.size
            x1 = max(0, int(b.get("x_min", 0) * w)); y1 = max(0, int(b.get("y_min", 0) * h))
            x2 = min(w, int(b.get("x_max", 1) * w)); y2 = min(h, int(b.get("y_max", 1) * h))
            if x2 - x1 > 4 and y2 - y1 > 4:
                img = img.crop((x1, y1, x2, y2))
        images.append(img)
    return images


# ---------------------------------------------------------------------------
# Metrics (mirrors EgoOwnershipBench.evaluate schema)
# ---------------------------------------------------------------------------

def compute_metrics(items, preds: list[str]) -> tuple[dict, pd.DataFrame]:
    gt = [it["label"] for it in items]
    n = len(gt)
    hit = [int(p == g) for p, g in zip(preds, gt)]
    results: dict = {
        "overall_acc": float(np.mean(hit)) if n else 0.0,
        "parsed_rate": 1.0,
        "n": n,
    }
    f1s = []
    for label in _LABELS:
        gt_mask = np.array([g == label for g in gt])
        pd_mask = np.array([p == label for p in preds])
        tp = int((gt_mask & pd_mask).sum())
        prec = tp / int(pd_mask.sum()) if pd_mask.any() else 0.0
        rec = tp / int(gt_mask.sum()) if gt_mask.any() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if gt_mask.any():
            f1s.append(f1)
        results[f"acc:{label}"] = float(np.array(hit)[gt_mask].mean()) if gt_mask.any() else None
        results[f"f1:{label}"] = round(f1, 4) if gt_mask.any() else None
        results[f"n:{label}"] = int(gt_mask.sum())
        results[f"pred_frac:{label}"] = float(pd_mask.mean()) if n else 0.0
    results["macro_f1"] = round(float(np.mean(f1s)), 4) if f1s else 0.0

    gt_abst = np.array([g == _ABSTAIN for g in gt])
    pd_abst = np.array([p == _ABSTAIN for p in preds])
    tp = int((gt_abst & pd_abst).sum())
    results["abstain_precision"] = round(tp / int(pd_abst.sum()), 4) if pd_abst.any() else None
    results["abstain_recall"] = round(tp / int(gt_abst.sum()), 4) if gt_abst.any() else None
    non = ~gt_abst
    results["over_abstention_rate"] = round(
        float((pd_abst & non).sum() / max(1, int(non.sum()))), 4
    )

    for col in ("taxonomy", "source_dataset"):
        vals = [it.get(col) for it in items]
        if any(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in vals):
            sub = pd.DataFrame({"k": vals, "hit": hit})
            for key, g in sub.groupby("k", dropna=False):
                results[f"acc:{col}={key}"] = float(g["hit"].mean())
                results[f"n:{col}={key}"] = int(len(g))

    cm = pd.DataFrame(0, index=_LABELS, columns=_LABELS, dtype=int)
    for g, p in zip(gt, preds):
        if p in _LABELS:
            cm.loc[g, p] += 1
    return results, cm


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def zeroshot_predict(embedder: Embedder, items, image_feats: np.ndarray) -> list[str]:
    """Per-item argmax over label prototypes; templates are noun-specific, so
    prototypes are built per unique noun and cached."""
    proto_cache: dict[str, np.ndarray] = {}

    def prototypes(noun: str) -> np.ndarray:
        if noun not in proto_cache:
            protos = []
            for label in _LABELS:
                texts = [t.format(noun=noun) for t in _ZS_TEMPLATES[label]]
                emb = embedder.embed_texts(texts).mean(axis=0)
                protos.append(emb / (np.linalg.norm(emb) + 1e-8))
            proto_cache[noun] = np.stack(protos)  # [4, D]
        return proto_cache[noun]

    preds = []
    for feat, it in zip(image_feats, items):
        sims = prototypes(it["noun"]) @ feat
        preds.append(_LABELS[int(np.argmax(sims))])
    return preds


def linear_probe_predict(
    items, feats: np.ndarray, folds: int, seed: int, *, group_by_video: bool = False
) -> list[str]:
    """Out-of-fold predictions from a multinomial logistic regression.

    ``group_by_video=True`` keeps all clips of one source video in the same
    fold (StratifiedGroupKFold) — the item-level split lets the probe exploit
    scene memorization (same kitchen/table in train and test), so the
    video-grouped number is the defensible one; report both.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    y = np.array([it["label"] for it in items])
    preds = np.empty(len(y), dtype=object)

    # Guard: stratified CV needs >= folds samples per class.
    counts = pd.Series(y).value_counts()
    eff_folds = max(2, min(folds, int(counts.min()))) if len(counts) > 1 else 0
    if eff_folds < 2:
        raise SystemExit(
            f"[probe] cannot stratify: label counts {counts.to_dict()} — "
            "need >=2 per class. Increase data or drop the missing class."
        )
    if eff_folds < folds:
        print(f"[probe] reducing folds {folds}->{eff_folds} (min class count)")

    if group_by_video:
        groups = np.array([str(it.get("video_id") or f"solo_{i}")
                           for i, it in enumerate(items)])
        n_groups = len(set(groups))
        eff_folds = min(eff_folds, n_groups)
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            splitter = StratifiedGroupKFold(
                n_splits=eff_folds, shuffle=True, random_state=seed
            )
        except ImportError:
            print("[probe] StratifiedGroupKFold unavailable — GroupKFold fallback")
            splitter = GroupKFold(n_splits=eff_folds)
        split_iter = splitter.split(feats, y, groups=groups)
    else:
        skf = StratifiedKFold(n_splits=eff_folds, shuffle=True, random_state=seed)
        split_iter = skf.split(feats, y)

    for tr, te in split_iter:
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(feats[tr], y[tr])
        preds[te] = clf.predict(feats[te])

    # Grouped splits can leave a class absent from a training fold; any item
    # never predicted (shouldn't happen with out-of-fold cover) is marked.
    preds = [p if isinstance(p, str) else "UNPREDICTED" for p in preds]
    return list(preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="EgoOwn",
                    help="EgoOwn | EgoOwn_EgoLife (base configs; probe uses frame t)")
    ap.add_argument("--model", default="clip",
                    help="clip | siglip | any HF id with get_image/text_features")
    ap.add_argument("--probe", default="both", choices=["zeroshot", "linear", "both"])
    ap.add_argument("--features", default="both", choices=["frame", "crop", "both"],
                    help="linear-probe features: frame t, bbox crop, or concat")
    ap.add_argument("--cv", default="both", choices=["item", "video", "both"],
                    help="linear-probe CV split: item-level (optimistic), "
                         "video-grouped (defensible), or both (report pair)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-dir", default="./outputs")
    args = ap.parse_args()

    model_id = _MODEL_ALIASES.get(args.model, args.model)
    probe_name = f"{args.model.replace('/', '_')}-probe"
    items, bench = load_items(args.dataset)
    if not items:
        raise SystemExit("[probe] no usable items (missing frames?)")

    embedder = Embedder(model_id, batch_size=args.batch_size)

    print(f"[probe] embedding full frames ({len(items)})...")
    frame_feats = embedder.embed_images(load_images(items, crop=False))
    crop_feats = None
    if args.features in ("crop", "both") and any(it["bbox"] for it in items):
        print("[probe] embedding bbox crops...")
        crop_feats = embedder.embed_images(load_images(items, crop=True))

    out_root = osp.join(args.out_dir, probe_name)
    os.makedirs(out_root, exist_ok=True)
    manifest_base = {
        "dataset": args.dataset,
        "model": model_id,
        "probe_version": PROBE_VERSION,
        "ref_field": bench.ref_field,
        "n_with_bbox": int(sum(1 for it in items if it["bbox"])),
        "eval_code_rev": _git_rev(osp.join(osp.dirname(__file__), "..")),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    def emit(kind: str, preds: list[str], extra: dict):
        results, cm = compute_metrics(items, preds)
        results["manifest"] = {**manifest_base, "mode": f"probe-{kind}", **extra}
        base = osp.join(out_root, f"{args.dataset}_probe_{kind}")
        with open(base + "_score.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        cm.to_csv(base + "_acc.csv")
        print(f"[probe:{kind}] acc={results['overall_acc']:.3f} "
              f"macroF1={results['macro_f1']:.3f} -> {base}_score.json")

    if args.probe in ("zeroshot", "both"):
        # Zero-shot on the most object-specific view available.
        zs_feats = crop_feats if crop_feats is not None else frame_feats
        preds = zeroshot_predict(embedder, items, zs_feats)
        emit("zeroshot", preds, {
            "zs_view": "crop" if crop_feats is not None else "frame",
            "templates_per_label": {k: len(v) for k, v in _ZS_TEMPLATES.items()},
        })

    if args.probe in ("linear", "both"):
        if args.features == "frame" or crop_feats is None:
            feats, feat_desc = frame_feats, "frame"
        elif args.features == "crop":
            feats, feat_desc = crop_feats, "crop"
        else:
            feats, feat_desc = np.concatenate([frame_feats, crop_feats], axis=1), "frame+crop"

        clf_desc = "LogisticRegression(C=1.0, max_iter=2000), out-of-fold preds"
        if args.cv in ("item", "both"):
            preds = linear_probe_predict(items, feats, args.folds, args.seed)
            emit("linear", preds, {
                "features": feat_desc, "folds": args.folds, "seed": args.seed,
                "cv": "item-level StratifiedKFold (optimistic: same-video clips "
                      "can appear in train and test — scene memorization possible)",
                "classifier": clf_desc,
            })
        if args.cv in ("video", "both"):
            preds = linear_probe_predict(
                items, feats, args.folds, args.seed, group_by_video=True
            )
            emit("linear_groupcv", preds, {
                "features": feat_desc, "folds": args.folds, "seed": args.seed,
                "cv": "video-grouped StratifiedGroupKFold (defensible: no video "
                      "spans train/test — this is the number to headline)",
                "classifier": clf_desc,
            })


if __name__ == "__main__":
    main()
