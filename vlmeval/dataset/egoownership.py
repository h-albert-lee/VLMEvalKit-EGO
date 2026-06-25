"""EgoOwnershipBench — Egocentric Implicit Ownership benchmark.

Wraps the HuggingFace dataset `Albertmade/ego-implicit-ownership-multiperson`
(3 parquet configs: `default`, `narration_a`, `egolife`) into a VLMEvalKit
4-class MCQ task.

Each clip is shown as up to 3 sparse frames (t-2, t-1, t) — when available —
plus a narration sentence + verb/noun metadata. The model picks one of
MINE / PERSON_k / SHARED / AMBIGUOUS for the salient object.

Reference labels are the Claude (`claude-jupiter-v1-p`) judgements stored in
`vlm_label`; this dataset has no human ground truth, so accuracy here measures
agreement with Claude. Override with EGOOWN_REF_FIELD=rule_label to compare
against the rule heuristic instead (only `egolife` populates that field).

Source layout (per the `default` parquet):
  hd_epic/<video_id>/<clip>__<tag>.jpg
  epic_kitchens/<video_id>/<clip>__<tag>.jpg
  egolife/<video_id>/<clip>__<tag>.jpg
"""

from __future__ import annotations

import os
import os.path as osp
import string
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load

from .image_base import ImageBaseDataset

_HF_REPO = "Albertmade/ego-implicit-ownership-multiperson"
_LABELS = ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS"]
_OPT_LETTERS = ["A", "B", "C", "D"]
_LABEL_BY_LETTER = dict(zip(_OPT_LETTERS, _LABELS))
_LETTER_BY_LABEL = {v: k for k, v in _LABEL_BY_LETTER.items()}

_CONFIG_BY_NAME = {
    # vlmeval dataset name -> (HF parquet filename relative to repo root, has-frames)
    # Filenames mirror the `configs:` block in the dataset README's YAML
    # frontmatter, so anyone re-pulling from HF gets the same files.
    "EgoOwn":         ("data/train.parquet",       True),
    "EgoOwn_NarrA":   ("data/narration_a.parquet", False),
    "EgoOwn_EgoLife": ("data/egolife.parquet",     True),
}

_TASK_INSTRUCTION = (
    "You are watching short first-person (egocentric) clips. For each clip we "
    "show up to 3 sparse frames (t-2, t-1, t) plus a narration describing what "
    "happens. Decide who owns the salient object referenced by the action.\n\n"
    "Label definitions:\n"
    "  MINE      — owned by the camera wearer (the person whose head the camera is on)\n"
    "  PERSON_k  — owned by another visible person\n"
    "  SHARED    — communal / table-center / not personally owned\n"
    "  AMBIGUOUS — symmetric, occluded, or insufficient evidence to decide\n"
)


def _frame_rel_path(frame_struct):
    """Pull `frame_path` out of the nested struct column, robustly."""
    if frame_struct is None:
        return None
    if isinstance(frame_struct, dict):
        return frame_struct.get("frame_path")
    # numpy void / structured record fallback
    try:
        return frame_struct["frame_path"]
    except Exception:
        return None


class EgoOwnershipBench(ImageBaseDataset):
    """4-way ownership classification over egocentric video clips."""

    TYPE = "MCQ"
    MODALITY = "IMAGE"
    DATASET_URL = {k: _HF_REPO for k in _CONFIG_BY_NAME}
    DATASET_MD5 = {}

    def __init__(self, dataset: str = "EgoOwn", **kwargs):
        if dataset not in _CONFIG_BY_NAME:
            raise ValueError(
                f"Unknown EgoOwnership config {dataset!r}; "
                f"expected one of {list(_CONFIG_BY_NAME)}"
            )
        self._parquet_filename, self._has_frames = _CONFIG_BY_NAME[dataset]
        # Skip ImageBaseDataset's tsv-download pipeline; load via _prepare()
        self.dataset_name = dataset
        ROOT = LMUDataRoot()
        self.img_root = osp.join(ROOT, "images", "EgoOwn")  # unused, we use absolute paths
        os.makedirs(self.img_root, exist_ok=True)

        self.data = self._prepare(dataset)
        self.meta_only = True  # build_prompt reads paths directly from `image_path`
        self.skip_noimg = False
        self.post_build(dataset)

    @classmethod
    def supported_datasets(cls):
        return list(_CONFIG_BY_NAME)

    # ----------------------------- data prep ---------------------------------

    def _resolve_local_root(self) -> str | None:
        """Optional local source: avoids HF download in dev. Set EGOOWN_LOCAL_ROOT
        to a directory that contains `data/<cfg>.parquet` and `frames/`.
        """
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
            repo_id=_HF_REPO,
            filename=parquet_filename,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )

    def _resolve_frame_root(self) -> str | None:
        """Return a local mirror root (if EGOOWN_LOCAL_ROOT is set), else None.

        When None, frames are pulled lazily via the HF hub cache.
        """
        local = self._resolve_local_root()
        if local is not None:
            cand = osp.join(local, "frames")
            if osp.isdir(cand):
                return cand
        return None

    def _ensure_frame(self, rel_path: str, frame_root: str | None) -> str | None:
        """Resolve a per-frame relative path (e.g. hd_epic/<vid>/<clip>__t.jpg)
        to an absolute local file; download from HF on miss. Returns None if
        unavailable (some clips in narration_a / a few egolife rows).
        """
        if not rel_path:
            return None
        if frame_root is not None:
            full = osp.join(frame_root, rel_path)
            if osp.exists(full):
                return full

        try:
            from huggingface_hub import hf_hub_download

            return hf_hub_download(
                repo_id=_HF_REPO,
                filename=f"frames/{rel_path}",
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[EgoOwn] missing frame {rel_path}: {exc}")
            return None

    def _prepare(self, dataset: str) -> pd.DataFrame:
        parquet_path = self._download_parquet(self._parquet_filename)
        df = pd.read_parquet(parquet_path)

        # Reference label for MCQ scoring.
        ref_field = os.environ.get("EGOOWN_REF_FIELD", "vlm_label")
        if ref_field not in df.columns:
            raise ValueError(
                f"Reference field {ref_field!r} missing from {self._parquet_filename} parquet; "
                f"available: {list(df.columns)}"
            )

        # Keep rows whose reference label is in our 4-class space.
        df = df[df[ref_field].isin(_LABELS)].reset_index(drop=True)
        if not len(df):
            raise RuntimeError(
                f"No rows with a valid {ref_field} value in {self._parquet_filename} parquet."
            )

        # Apply EGOOWN_LIMIT before frame downloads so smoke tests don't pull
        # 1500+ JPGs from HF just to evaluate a handful.
        limit = int(os.environ.get("EGOOWN_LIMIT", "0") or 0)
        if limit and limit < len(df):
            df = df.iloc[:limit].reset_index(drop=True)
            print(f"[EgoOwn:{dataset}] EGOOWN_LIMIT={limit} -> {len(df)} rows")

        # Resolve frame paths once at load time (only configs with frames).
        if self._has_frames:
            from concurrent.futures import ThreadPoolExecutor

            frame_root = self._resolve_frame_root()
            tags = ("frame_t_minus_2", "frame_t_minus_1", "frame_t")

            # Build a unique work-list of relative paths to fetch.
            rels_per_row: list[list[str | None]] = []
            unique_rels: list[str] = []
            seen: set[str] = set()
            for i in range(len(df)):
                row_rels = []
                for tag in tags:
                    rel = _frame_rel_path(df.iloc[i][tag])
                    row_rels.append(rel)
                    if rel and rel not in seen:
                        seen.add(rel)
                        unique_rels.append(rel)
                rels_per_row.append(row_rels)

            workers = int(os.environ.get("EGOOWN_DL_WORKERS", "16"))
            print(
                f"[EgoOwn:{dataset}] downloading/resolving {len(unique_rels)} frames "
                f"with {workers} workers..."
            )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                resolved = list(
                    pool.map(lambda r: (r, self._ensure_frame(r, frame_root)), unique_rels)
                )
            path_by_rel: dict[str, str] = {r: p for r, p in resolved if p}

            paths_per_row = []
            missing_counts = defaultdict(int)
            for row_rels in rels_per_row:
                row_paths: list[str] = []
                for rel in row_rels:
                    if rel is None:
                        missing_counts["no_rel"] += 1
                        continue
                    p = path_by_rel.get(rel)
                    if p:
                        row_paths.append(p)
                    else:
                        missing_counts["no_file"] += 1
                paths_per_row.append(row_paths)
            df["image_path"] = paths_per_row
            n_full = sum(1 for p in paths_per_row if len(p) == 3)
            print(
                f"[EgoOwn:{dataset}] frames resolved: {n_full}/{len(df)} clips with all 3, "
                f"missing rel={missing_counts['no_rel']} file={missing_counts['no_file']}"
            )
        else:
            df["image_path"] = [[] for _ in range(len(df))]

        # MCQ scaffolding: question + A/B/C/D options + reference answer.
        df["question"] = "Who owns the salient object referenced by this action?"
        for letter, label in _LABEL_BY_LETTER.items():
            df[letter] = label
        df["answer"] = df[ref_field].map(_LETTER_BY_LABEL)

        # Stable string index (required by VLMEvalKit).
        df["index"] = df["clip_id"].astype(str)

        # Carry diagnostic fields for evaluate()
        keep = [
            "index", "clip_id", "source_dataset", "video_id", "taxonomy",
            "verb", "narration", "image_path",
            "question", *_OPT_LETTERS, "answer",
            ref_field, "rule_label",
        ]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].copy()
        df.attrs["ref_field"] = ref_field
        return df

    # --------------------------- prompt + dump -------------------------------

    def dump_image(self, line):
        # `image_path` is already a list of absolute local paths.
        paths = line["image_path"]
        if isinstance(paths, np.ndarray):
            paths = list(paths)
        if not isinstance(paths, (list, tuple)):
            paths = [paths] if paths else []
        return list(paths)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        paths = self.dump_image(line)

        narration = line.get("narration") if isinstance(line, dict) else line["narration"]
        verb = line.get("verb") if isinstance(line, dict) else line["verb"]
        taxonomy = line.get("taxonomy") if isinstance(line, dict) else line["taxonomy"]
        src_dataset = line.get("source_dataset") if isinstance(line, dict) else line["source_dataset"]

        narration = narration if (isinstance(narration, str) and narration) else "—"
        verb = verb if (isinstance(verb, str) and verb) else "—"

        opts_block = "\n".join(f"{l}. {_LABEL_BY_LETTER[l]}" for l in _OPT_LETTERS)

        if paths:
            frame_caption = (
                f"Frames shown above are t-2, t-1, t (chronological) from "
                f"the clip below.\n"
            )
        else:
            frame_caption = "No frames available; decide from narration text alone.\n"

        prompt = (
            f"{_TASK_INSTRUCTION}\n"
            f"{frame_caption}\n"
            f"Clip metadata\n"
            f"  source_dataset: {src_dataset}\n"
            f"  taxonomy: {taxonomy}\n"
            f"  verb: {verb}\n"
            f"  narration: {narration}\n\n"
            f"Question: {line['question']}\n"
            f"Options:\n{opts_block}\n\n"
            "Respond with a single capital letter (A, B, C, or D) corresponding "
            "to the correct label."
        )

        msgs = [dict(type="image", value=p) for p in paths]
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
        # 1) leading single letter "A", "A.", "A)" ...
        head = s.lstrip().lstrip("(").lstrip("[")
        if head and head[0].upper() in _OPT_LETTERS:
            nxt = head[1:2]
            if nxt in {"", ".", ")", "]", ":", " ", "\n", ",", "-", "/"}:
                return head[0].upper()
        # 2) explicit label name anywhere in the text (longest first)
        up = s.upper()
        for label in sorted(_LABELS, key=len, reverse=True):
            if label in up:
                return _LETTER_BY_LABEL[label]
        # 3) bare uppercase letter token
        import re
        m = re.search(r"\b([A-D])\b", s)
        if m:
            return m.group(1)
        return None

    def evaluate(self, eval_file, **judge_kwargs):
        score_file = get_intermediate_file_path(eval_file, "_acc", "csv")
        report_file = get_intermediate_file_path(eval_file, "_score", "json")

        data = load(eval_file)
        if "answer" not in data.columns or "prediction" not in data.columns:
            raise ValueError("eval_file must contain `answer` and `prediction` columns")

        # Re-attach metadata fields from the meta df for breakdowns.
        meta_cols = [c for c in ("taxonomy", "source_dataset", "rule_label") if c in self.data.columns]
        if meta_cols:
            meta = self.data[["index"] + meta_cols].copy()
            meta["index"] = meta["index"].astype(str)
            data["index"] = data["index"].astype(str)
            data = data.merge(meta, on="index", how="left", suffixes=("", "_meta"))

        # Predicted letter & label.
        pred_letters = [self._extract_letter(p) for p in data["prediction"]]
        data["pred_letter"] = pred_letters
        data["pred_label"] = [
            _LABEL_BY_LETTER[l] if l in _LABEL_BY_LETTER else None for l in pred_letters
        ]
        data["gt_label"] = [_LABEL_BY_LETTER.get(str(a), str(a)) for a in data["answer"]]
        data["hit"] = [
            int(pl == gl) if pl is not None else 0
            for pl, gl in zip(data["pred_label"], data["gt_label"])
        ]
        data["parsed"] = [int(p is not None) for p in pred_letters]

        # Breakdowns.
        results: dict[str, float] = {
            "overall_acc": float(data["hit"].mean()) if len(data) else 0.0,
            "parsed_rate": float(data["parsed"].mean()) if len(data) else 0.0,
            "n": int(len(data)),
        }

        # Per-label (gt) accuracy.
        for label in _LABELS:
            mask = data["gt_label"] == label
            results[f"acc:{label}"] = float(data.loc[mask, "hit"].mean()) if mask.any() else None
            results[f"n:{label}"] = int(mask.sum())

        # Per-taxonomy / per-source dataset.
        for col in ("taxonomy", "source_dataset"):
            if col in data.columns:
                for key, sub in data.groupby(col, dropna=False):
                    results[f"acc:{col}={key}"] = float(sub["hit"].mean())
                    results[f"n:{col}={key}"] = int(len(sub))

        # Confusion matrix (gt × pred, including "UNPARSED").
        labels_plus = _LABELS + ["UNPARSED"]
        cm = pd.DataFrame(0, index=labels_plus, columns=labels_plus, dtype=int)
        for gl, pl in zip(data["gt_label"], data["pred_label"]):
            cm.loc[gl, pl if pl in _LABELS else "UNPARSED"] += 1

        # Save artefacts.
        scored_path = get_intermediate_file_path(eval_file, "_scored", "xlsx")
        dump(data, scored_path)
        dump(cm, score_file)            # CSV: confusion matrix
        dump(results, report_file)      # JSON: scalar metrics

        return results
