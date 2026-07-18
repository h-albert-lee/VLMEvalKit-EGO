"""EgoOwnershipBench — Egocentric Implicit Ownership benchmark.

Wraps the HuggingFace dataset `Albertmade/ego-implicit-ownership-multiperson`
into a VLMEvalKit 4-class MCQ task with **four input modes**, selected by the
dataset-name suffix:

  EgoOwn                sparse-frame  (t-2, t-1, t)          [paper setting (b)]
  EgoOwn_Single         single-frame  (t only)               [paper setting (a)]
  EgoOwn_Clip           short clip    (dense frames / mp4 from source video)
  EgoOwn_Blind          text-only     (no images; narration + metadata)
  EgoOwn_EgoLife[...]   same modes over the egolife parquet
  EgoOwn_NarrA          legacy narration-only parquet (kept for compatibility)

Design invariants (paper §3 / §5, labeling guideline v2-2026-07-04):
  * Prompt encodes guideline v2: ownership ≠ possession + boundary rules.
  * NO diagnostic leakage: taxonomy / source_dataset are never shown to the
    model; narration is shown only in Blind mode (or when
    EGOOWN_INCLUDE_NARRATION=1 is explicitly set for ablations).
  * Option order is shuffled per item, deterministically from
    (clip_id, EGOOWN_OPT_SEED). Re-running with a different EGOOWN_OPT_SEED
    yields a new-but-reproducible permutation — this is the §5.4
    option-order permutation test.
  * evaluate() emits per-label acc, macro-F1, per-taxonomy acc, abstention
    precision/recall, over-abstention, prediction distribution, confusion
    matrix, and a reproducibility manifest (prompt version, seeds, git rev,
    ref field, mode).

Reference labels default to `vlm_label` (Claude judge); once the human
re-review lands, set EGOOWN_REF_FIELD=human_label (or rule_label for the
cascade heuristic).

Env knobs:
  EGOOWN_REF_FIELD           reference label column   (default: vlm_label)
  EGOOWN_OPT_SEED            option-shuffle seed      (default: 0)
  EGOOWN_LIMIT               row limit for smoke runs (default: 0 = all)
  EGOOWN_LOCAL_ROOT          local mirror with data/*.parquet and frames/
  EGOOWN_DL_WORKERS          parallel frame downloads (default: 16)
  EGOOWN_INCLUDE_NARRATION   =1 leaks narration into visual modes (ablation)
  EGOOWN_VIDEOS_ROOT         {video_id}.mp4 dir — required for Clip mode
  EGOOWN_CLIP_NFRAMES        dense frames per clip    (default: 8)
  EGOOWN_CLIP_AS_VIDEO       =1 pass an mp4 (video-native models) instead of
                             dense frames
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import random
import subprocess
import warnings
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load

from .image_base import ImageBaseDataset

_HF_REPO = "ego-ownership/merged-to-review"  # final benchmark (gated; needs HF_TOKEN)
_LABELS = ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS"]
_OPT_LETTERS = ["A", "B", "C", "D"]
_ABSTAIN_LABEL = "AMBIGUOUS"

PROMPT_VERSION = "v2-2026-07-04"

_MODE_SUFFIX = {"_Single": "single", "_Clip": "clip", "_Blind": "blind"}

# base config -> (parquet filename, has frame columns)
# 2026-07-16: switched to the final merged benchmark (3,229 rows, 3 sources
# unified; per-source breakdown via the source_dataset column). Legacy
# EgoOwn_EgoLife / EgoOwn_NarrA configs removed — no counterpart files in the
# new repo.
_BASE_CONFIGS = {
    "EgoOwn": ("data/eval.parquet", True),
}


def _parse_dataset_name(dataset: str) -> tuple[str, str]:
    """Return (base_config, mode). Default mode is sparse."""
    for suffix, mode in _MODE_SUFFIX.items():
        if dataset.endswith(suffix):
            base = dataset[: -len(suffix)]
            if base in _BASE_CONFIGS:
                return base, mode
    if dataset in _BASE_CONFIGS:
        return dataset, "sparse"
    raise ValueError(
        f"Unknown EgoOwnership dataset {dataset!r}; expected one of "
        f"{sorted(_all_dataset_names())}"
    )


def _all_dataset_names() -> list[str]:
    names = []
    for base in _BASE_CONFIGS:
        names.append(base)  # sparse default
        names.extend(base + s for s in _MODE_SUFFIX)
    return names


# ---------------------------------------------------------------------------
# Prompt (guideline v2) — keep in sync with label-pipeline vlm_crosscheck.py
# ---------------------------------------------------------------------------

_TASK_INSTRUCTION = """\
You are shown evidence from a short first-person (egocentric) video. Decide \
who OWNS the salient object involved in the action. Ownership is not the \
same as possession: who is holding or using the object right now does not by \
itself decide whose it is.

Label definitions:
{label_defs}

Apply these boundary rules — they OVERRIDE naive proximity or possession cues:
  1. Place-setting: tableware set at a person's seat belongs to that person
     even before they touch it (setting the place counts as first possession).
  2. Communal persistence: an inherently communal item (serving spoon, shared
     bottle, serving platter, communal dish) stays SHARED even while one
     person is holding or using it — transient use does not transfer ownership.
  3. Function-first ambiguity: an inherently communal-function item defaults
     to SHARED; a personal-function item (cup, phone, pen) with no attribution
     cues is AMBIGUOUS.
  4. Holding is not owning: an object in someone's hand is not automatically
     theirs — infer whose it is, not who controls it right now.
  5. Abandonment: pushing one's own item into shared space does not transfer
     or void ownership.
"""

_LABEL_DEF_LINES = {
    "MINE": "the object belongs to the camera wearer (ego)",
    "PERSON_k": "the object belongs to another person in the scene",
    "SHARED": (
        "a communal item that belongs to no single person "
        "(inherently shared by function, or by established shared use)"
    ),
    "AMBIGUOUS": (
        "the visual evidence is insufficient to attribute ownership to anyone"
    ),
}

_FRAME_CAPTIONS = {
    "sparse": (
        "The images above are three sparse frames from the clip in "
        "chronological order: t-2 (2s before the action), t-1 (1s before), "
        "and t (the action moment). Judge ownership AT frame t, using the "
        "earlier frames as context."
    ),
    "single": (
        "The image above is the action-moment frame (t) of the clip. Judge "
        "ownership at this moment."
    ),
    "clip": (
        "You are shown the clip (or densely sampled frames from it) in "
        "chronological order, ending at the action moment t. Judge ownership "
        "AT the final moment, using the earlier footage as context."
    ),
    "blind": (
        "No images are available. Decide from the text description alone; "
        "if the text is insufficient to attribute ownership, answer with the "
        "insufficient-evidence option."
    ),
}


def _option_order_for(clip_id: str, seed: int) -> list[str]:
    """Deterministic per-item label order (varies across items and seeds)."""
    digest = hashlib.sha256(f"{seed}:{clip_id}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    order = list(_LABELS)
    rng.shuffle(order)
    return order


def _frame_rel_path(frame_struct):
    if frame_struct is None:
        return None
    if isinstance(frame_struct, dict):
        return frame_struct.get("frame_path")
    try:
        return frame_struct["frame_path"]
    except Exception:
        return None


def _git_rev(repo_dir: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class EgoOwnershipBench(ImageBaseDataset):
    """4-way ownership classification over egocentric video clips."""

    TYPE = "MCQ"
    MODALITY = "IMAGE"
    DATASET_URL = {k: _HF_REPO for k in _all_dataset_names()}
    DATASET_MD5 = {}

    def __init__(self, dataset: str = "EgoOwn", **kwargs):
        base, mode = _parse_dataset_name(dataset)
        self._base_config = base
        self.mode = mode
        self._parquet_filename, self._has_frames = _BASE_CONFIGS[base]
        self.opt_seed = int(os.environ.get("EGOOWN_OPT_SEED", "0") or 0)
        # Default: human-audited final GT. vlm_label exists only for comparison
        # (and is null for egolife/rescue rows — do NOT use it as reference).
        self.ref_field = os.environ.get("EGOOWN_REF_FIELD", "human_label")
        self.include_narration = (
            mode == "blind" or os.environ.get("EGOOWN_INCLUDE_NARRATION") == "1"
        )

        self.dataset_name = dataset
        ROOT = LMUDataRoot()
        self.img_root = osp.join(ROOT, "images", "EgoOwn")
        os.makedirs(self.img_root, exist_ok=True)

        self.data = self._prepare(dataset)
        self.meta_only = True
        self.skip_noimg = False
        self.post_build(dataset)

    @classmethod
    def supported_datasets(cls):
        return _all_dataset_names()

    # ----------------------------- data prep ---------------------------------

    def _resolve_local_root(self) -> str | None:
        local = os.environ.get("EGOOWN_LOCAL_ROOT")
        if local and osp.isdir(local):
            return local
        return None

    def _download_parquet(self, parquet_filename: str) -> str:
        local = self._resolve_local_root()
        if local is not None:
            cand = osp.join(local, parquet_filename)
            if osp.exists(cand):
                return cand
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=_HF_REPO, filename=parquet_filename, repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )

    def _resolve_frame_root(self) -> str | None:
        local = self._resolve_local_root()
        if local is not None:
            cand = osp.join(local, "frames")
            if osp.isdir(cand):
                return cand
        return None

    def _ensure_frame(self, rel_path: str, frame_root: str | None) -> str | None:
        if not rel_path:
            return None
        if frame_root is not None:
            full = osp.join(frame_root, rel_path)
            if osp.exists(full):
                return full
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(
                repo_id=_HF_REPO, filename=f"frames/{rel_path}",
                repo_type="dataset", token=os.environ.get("HF_TOKEN"),
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[EgoOwn] missing frame {rel_path}: {exc}")
            return None

    # ---- frame reconstruction (metadata-only rows, e.g. Ego4D) ----

    def _reconstruct_frame(self, row, tag: str) -> str | None:
        """Extract one sparse frame from the source video for rows whose
        frame_path is null (Ego4D metadata-only distribution).

        Needs EGOOWN_VIDEOS_ROOT/{video_id}.mp4 and the frame struct's
        timestamp_sec. Timestamps are treated as absolute in the source
        video; set EGOOWN_TS_MODE=clip_relative if the parquet stores
        clip-relative times (then source_video_start_sec is added).
        Verify visually on a smoke run before real sweeps.
        """
        videos_root = os.environ.get("EGOOWN_VIDEOS_ROOT")
        if not videos_root:
            return None
        struct = row.get(tag) if tag in row.index else None
        ts = None
        if isinstance(struct, dict):
            ts = struct.get("timestamp_sec")
        else:
            try:
                ts = struct["timestamp_sec"]
            except Exception:
                ts = None
        video_id = row.get("video_id")
        if ts is None or not video_id:
            return None
        if os.environ.get("EGOOWN_TS_MODE") == "clip_relative":
            start = row.get("source_video_start_sec")
            if start is not None and not (isinstance(start, float) and np.isnan(start)):
                ts = float(start) + float(ts)
        video_path = osp.join(videos_root, f"{video_id}.mp4")
        if not osp.exists(video_path):
            return None
        cache = osp.join(LMUDataRoot(), "egoown_recon")
        os.makedirs(cache, exist_ok=True)
        safe_id = str(row["index"]).replace("/", "_").replace("#", "__")
        dest = osp.join(cache, f"{safe_id}__{tag}.jpg")
        if not osp.exists(dest):
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-ss", f"{max(0.0, float(ts)):.3f}", "-i", video_path,
                 "-frames:v", "1", "-q:v", "2", "-y", dest],
                check=False,
            )
        return dest if osp.exists(dest) else None

    # ---- clip mode helpers ----

    def _clip_media_for_row(self, row) -> tuple[list[str], str | None]:
        """Return (dense_frame_paths, mp4_path) for Clip mode.

        Requires EGOOWN_VIDEOS_ROOT with {video_id}.mp4 plus per-row timing
        columns (source_video_start_sec, frame_times_sec). Extraction is
        cached under LMUDataRoot()/egoown_clips.
        """
        videos_root = os.environ.get("EGOOWN_VIDEOS_ROOT")
        if not videos_root:
            raise RuntimeError(
                "Clip mode needs EGOOWN_VIDEOS_ROOT pointing at a directory "
                "with {video_id}.mp4 source videos."
            )
        video_id = row.get("video_id")
        start = row.get("source_video_start_sec")
        frame_times = row.get("frame_times_sec")
        if isinstance(frame_times, str):
            try:
                frame_times = json.loads(frame_times)
            except json.JSONDecodeError:
                frame_times = None
        if not video_id or start is None or not isinstance(frame_times, dict):
            raise RuntimeError(
                "Clip mode needs per-row `video_id`, `source_video_start_sec` "
                "and `frame_times_sec` columns in the parquet; this parquet "
                "revision lacks them — regenerate the dataset export or use "
                "Sparse/Single mode."
            )
        video_path = osp.join(videos_root, f"{video_id}.mp4")
        if not osp.exists(video_path):
            raise RuntimeError(f"Clip mode: source video not found: {video_path}")

        t_start = float(start) + float(min(frame_times.values()))
        t_end = float(start) + float(max(frame_times.values()))
        duration = max(0.5, t_end - t_start)

        cache = osp.join(LMUDataRoot(), "egoown_clips")
        os.makedirs(cache, exist_ok=True)
        safe_id = str(row["index"]).replace("/", "_").replace("#", "__")

        as_video = os.environ.get("EGOOWN_CLIP_AS_VIDEO") == "1"
        if as_video:
            mp4 = osp.join(cache, f"{safe_id}.mp4")
            if not osp.exists(mp4):
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error",
                     "-ss", f"{max(0.0, t_start):.3f}", "-t", f"{duration:.3f}",
                     "-i", video_path, "-c:v", "libx264", "-an", "-y", mp4],
                    check=True,
                )
            return [], mp4

        nframes = int(os.environ.get("EGOOWN_CLIP_NFRAMES", "8"))
        paths = []
        for i in range(nframes):
            ts = t_start + duration * i / max(1, nframes - 1)
            dest = osp.join(cache, f"{safe_id}__f{i:02d}.jpg")
            if not osp.exists(dest):
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error",
                     "-ss", f"{max(0.0, ts):.3f}", "-i", video_path,
                     "-frames:v", "1", "-q:v", "2", "-y", dest],
                    check=False,
                )
            if osp.exists(dest):
                paths.append(dest)
        return paths, None

    def _prepare(self, dataset: str) -> pd.DataFrame:
        parquet_path = self._download_parquet(self._parquet_filename)
        df = pd.read_parquet(parquet_path)

        ref_field = self.ref_field
        if ref_field not in df.columns:
            raise ValueError(
                f"Reference field {ref_field!r} missing from "
                f"{self._parquet_filename}; available: {list(df.columns)}"
            )
        df = df[df[ref_field].isin(_LABELS)].reset_index(drop=True)
        if not len(df):
            raise RuntimeError(
                f"No rows with a valid {ref_field} in {self._parquet_filename}."
            )

        limit = int(os.environ.get("EGOOWN_LIMIT", "0") or 0)
        if limit and limit < len(df):
            df = df.iloc[:limit].reset_index(drop=True)
            print(f"[EgoOwn:{dataset}] EGOOWN_LIMIT={limit} -> {len(df)} rows")

        df["index"] = df["clip_id"].astype(str)

        # ---- resolve frames (visual frame modes only) ----
        if self._has_frames and self.mode in ("sparse", "single"):
            from concurrent.futures import ThreadPoolExecutor

            frame_root = self._resolve_frame_root()
            tags = (
                ("frame_t",) if self.mode == "single"
                else ("frame_t_minus_2", "frame_t_minus_1", "frame_t")
            )
            rels_per_row: list[list[str | None]] = []
            unique_rels: list[str] = []
            seen: set[str] = set()
            for i in range(len(df)):
                row_rels = []
                for tag in tags:
                    rel = _frame_rel_path(df.iloc[i][tag]) if tag in df.columns else None
                    row_rels.append(rel)
                    if rel and rel not in seen:
                        seen.add(rel)
                        unique_rels.append(rel)
                rels_per_row.append(row_rels)

            workers = int(os.environ.get("EGOOWN_DL_WORKERS", "16"))
            print(
                f"[EgoOwn:{dataset}] resolving {len(unique_rels)} frames "
                f"({self.mode} mode) with {workers} workers..."
            )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                resolved = list(
                    pool.map(lambda r: (r, self._ensure_frame(r, frame_root)), unique_rels)
                )
            path_by_rel = {r: p for r, p in resolved if p}

            paths_per_row, missing = [], defaultdict(int)
            for i, row_rels in enumerate(rels_per_row):
                row_paths = []
                for tag, rel in zip(tags, row_rels):
                    p = path_by_rel.get(rel) if rel else None
                    if p is None:
                        # Metadata-only rows (Ego4D: license forbids frame
                        # redistribution) → reconstruct from the source video.
                        p = self._reconstruct_frame(df.iloc[i], tag)
                    if p:
                        row_paths.append(p)
                    else:
                        missing["no_rel" if rel is None else "no_file"] += 1
                paths_per_row.append(row_paths)
            df["image_path"] = paths_per_row
            want = len(tags)
            n_full = sum(1 for p in paths_per_row if len(p) == want)
            print(
                f"[EgoOwn:{dataset}] frames: {n_full}/{len(df)} rows complete "
                f"({want} frame(s) each); missing rel={missing['no_rel']} "
                f"file={missing['no_file']}"
            )

            # Guard against silent text-only contamination: in visual modes a
            # row with zero frames must NOT be evaluated as if it were seen.
            if os.environ.get("EGOOWN_ALLOW_NOIMG") != "1":
                has_img = df["image_path"].map(len) > 0
                n_drop = int((~has_img).sum())
                if n_drop:
                    by_src = df.loc[~has_img, "source_dataset"].value_counts().to_dict() \
                        if "source_dataset" in df.columns else {}
                    print(
                        f"[EgoOwn:{dataset}] EXCLUDING {n_drop} rows with no "
                        f"frames in {self.mode} mode {by_src} — set "
                        f"EGOOWN_VIDEOS_ROOT to reconstruct (Ego4D) or "
                        f"EGOOWN_ALLOW_NOIMG=1 to force-keep (NOT for real runs)."
                    )
                    df = df[has_img].reset_index(drop=True)
        else:
            df["image_path"] = [[] for _ in range(len(df))]

        # ---- per-item shuffled MCQ options ----
        answers, orders = [], []
        for i in range(len(df)):
            order = _option_order_for(df.iloc[i]["index"], self.opt_seed)
            orders.append(order)
            gt_label = df.iloc[i][ref_field]
            answers.append(_OPT_LETTERS[order.index(gt_label)])
        for j, letter in enumerate(_OPT_LETTERS):
            df[letter] = [order[j] for order in orders]
        df["answer"] = answers
        df["answer_label"] = df[ref_field]
        df["option_order"] = ["|".join(o) for o in orders]
        df["question"] = "Who owns the salient object involved in this action?"

        keep = [
            "index", "clip_id", "source_dataset", "video_id", "taxonomy",
            "verb", "narration", "image_path",
            "source_video_start_sec", "frame_times_sec", "guideline_version",
            "question", *_OPT_LETTERS, "answer", "answer_label", "option_order",
            ref_field, "rule_label",
        ]
        keep = [c for c in keep if c in df.columns]
        df = df[list(dict.fromkeys(keep))].copy()
        df.attrs["ref_field"] = ref_field
        return df

    # --------------------------- prompt + dump -------------------------------

    def dump_image(self, line):
        paths = line["image_path"]
        if isinstance(paths, np.ndarray):
            paths = list(paths)
        if not isinstance(paths, (list, tuple)):
            paths = [paths] if paths else []
        return list(paths)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        mp4_path = None
        if self.mode == "blind":
            paths = []
        elif self.mode == "clip":
            paths, mp4_path = self._clip_media_for_row(line)
        else:
            paths = self.dump_image(line)

        # Per-item option block (shuffled order fixed at prepare time).
        opts_block = "\n".join(
            f"{letter}. {line[letter]}" for letter in _OPT_LETTERS
        )
        label_defs = "\n".join(
            f"  {line[letter]:<9} — {_LABEL_DEF_LINES[line[letter]]}"
            for letter in _OPT_LETTERS
        )

        # Text evidence: only in blind mode (or explicit ablation flag) —
        # narration is a known textual shortcut (§5.4); taxonomy and
        # source_dataset are diagnostic metadata and are NEVER shown.
        evidence_lines = []
        if self.include_narration:
            narration = line.get("narration") if isinstance(line, dict) else line["narration"]
            verb = line.get("verb") if isinstance(line, dict) else line["verb"]
            narration = narration if isinstance(narration, str) and narration else "—"
            verb = verb if isinstance(verb, str) and verb else "—"
            evidence_lines += [
                "Text evidence:",
                f"  action verb: {verb}",
                f"  narration:   {narration}",
                "",
            ]

        prompt = (
            _TASK_INSTRUCTION.format(label_defs=label_defs)
            + "\n"
            + _FRAME_CAPTIONS[self.mode]
            + "\n\n"
            + "\n".join(evidence_lines)
            + f"Question: {line['question']}\n"
            + f"Options:\n{opts_block}\n\n"
            + "Respond with a single capital letter (A, B, C, or D)."
        )

        msgs = []
        if mp4_path:
            msgs.append(dict(type="video", value=mp4_path))
        msgs.extend(dict(type="image", value=p) for p in paths)
        msgs.append(dict(type="text", value=prompt))
        return msgs

    # ------------------------------ eval -------------------------------------

    @staticmethod
    def _extract_letter(prediction: str) -> str | None:
        if prediction is None:
            return None
        s = str(prediction).strip()
        if not s:
            return None
        head = s.lstrip().lstrip("(").lstrip("[")
        if head and head[0].upper() in _OPT_LETTERS:
            nxt = head[1:2]
            if nxt in {"", ".", ")", "]", ":", " ", "\n", ",", "-", "/"}:
                return head[0].upper()
        import re
        m = re.search(r"\b([A-D])\b", s)
        if m:
            return m.group(1)
        return None

    @classmethod
    def _extract_pred_label(cls, prediction: str, row) -> str | None:
        """Letter first (decoded via the row's shuffled mapping), else a label
        name mentioned anywhere in the response text."""
        letter = cls._extract_letter(prediction)
        if letter is not None:
            return row[letter]
        up = str(prediction).upper() if prediction is not None else ""
        for label in sorted(_LABELS, key=len, reverse=True):
            if label.upper() in up:
                return label
        return None

    def evaluate(self, eval_file, **judge_kwargs):
        score_file = get_intermediate_file_path(eval_file, "_acc", "csv")
        report_file = get_intermediate_file_path(eval_file, "_score", "json")

        data = load(eval_file)
        if "prediction" not in data.columns:
            raise ValueError("eval_file must contain a `prediction` column")

        # Re-attach meta + per-row option mapping from self.data (authoritative).
        meta_cols = [
            c for c in (
                "taxonomy", "source_dataset", "rule_label", "answer_label",
                "option_order", *_OPT_LETTERS,
            ) if c in self.data.columns
        ]
        meta = self.data[["index"] + meta_cols].copy()
        meta["index"] = meta["index"].astype(str)
        data["index"] = data["index"].astype(str)
        drop_dup = [c for c in meta_cols if c in data.columns]
        data = data.drop(columns=drop_dup).merge(meta, on="index", how="left")

        data["pred_label"] = [
            self._extract_pred_label(p, row)
            for p, (_, row) in zip(data["prediction"], data.iterrows())
        ]
        data["gt_label"] = data["answer_label"]
        data["hit"] = [
            int(pl == gl) if pl is not None else 0
            for pl, gl in zip(data["pred_label"], data["gt_label"])
        ]
        data["parsed"] = [int(pl is not None) for pl in data["pred_label"]]

        results: dict = {
            "overall_acc": float(data["hit"].mean()) if len(data) else 0.0,
            "parsed_rate": float(data["parsed"].mean()) if len(data) else 0.0,
            "n": int(len(data)),
        }

        # Per-label accuracy + per-label F1 -> macro-F1.
        f1s = []
        for label in _LABELS:
            gt_mask = data["gt_label"] == label
            pd_mask = data["pred_label"] == label
            tp = int((gt_mask & pd_mask).sum())
            prec = tp / int(pd_mask.sum()) if pd_mask.any() else 0.0
            rec = tp / int(gt_mask.sum()) if gt_mask.any() else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if gt_mask.any():
                f1s.append(f1)
            results[f"acc:{label}"] = float(data.loc[gt_mask, "hit"].mean()) if gt_mask.any() else None
            results[f"f1:{label}"] = round(f1, 4) if gt_mask.any() else None
            results[f"n:{label}"] = int(gt_mask.sum())
            # Prediction distribution — label-collapse / prior diagnostics.
            results[f"pred_frac:{label}"] = float(pd_mask.mean()) if len(data) else 0.0
        results["macro_f1"] = round(float(np.mean(f1s)), 4) if f1s else 0.0

        # Abstention diagnostics (AMBIGUOUS as abstain).
        gt_abst = data["gt_label"] == _ABSTAIN_LABEL
        pd_abst = data["pred_label"] == _ABSTAIN_LABEL
        tp = int((gt_abst & pd_abst).sum())
        results["abstain_precision"] = round(tp / int(pd_abst.sum()), 4) if pd_abst.any() else None
        results["abstain_recall"] = round(tp / int(gt_abst.sum()), 4) if gt_abst.any() else None
        non_abst = ~gt_abst
        results["over_abstention_rate"] = round(
            float((pd_abst & non_abst).sum() / max(1, int(non_abst.sum()))), 4
        )

        # Per-taxonomy / per-source breakdowns.
        for col in ("taxonomy", "source_dataset"):
            if col in data.columns:
                for key, sub in data.groupby(col, dropna=False):
                    results[f"acc:{col}={key}"] = float(sub["hit"].mean())
                    results[f"n:{col}={key}"] = int(len(sub))

        # Confusion matrix (gt × pred, incl. UNPARSED).
        labels_plus = _LABELS + ["UNPARSED"]
        cm = pd.DataFrame(0, index=labels_plus, columns=labels_plus, dtype=int)
        for gl, pl in zip(data["gt_label"], data["pred_label"]):
            cm.loc[gl, pl if pl in _LABELS else "UNPARSED"] += 1

        # Reproducibility manifest.
        results["manifest"] = {
            "dataset": self.dataset_name,
            "base_config": self._base_config,
            "mode": self.mode,
            "prompt_version": PROMPT_VERSION,
            "opt_seed": self.opt_seed,
            "ref_field": self.ref_field,
            "include_narration": self.include_narration,
            "eval_code_rev": _git_rev(osp.dirname(osp.dirname(osp.dirname(__file__)))),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "eval_file": osp.basename(str(eval_file)),
        }

        scored_path = get_intermediate_file_path(eval_file, "_scored", "xlsx")
        dump(data, scored_path)
        dump(cm, score_file)
        dump(results, report_file)
        return results
