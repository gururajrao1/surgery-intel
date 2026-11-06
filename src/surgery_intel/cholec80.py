"""Real Cholec80 layout support.

Layout (verified against the official archive + community loaders):
  cholec80/
    videos/video01.mp4 ...                    25 fps
    phase_annotations/video01-phase.txt ...   "Frame<TAB>Phase", 0-based, 25 fps
    tool_annotations/video01-tool.txt ...     per-frame binary tool presence, 1 fps

Standard protocol: videos 01-40 train, 41-80 test, 1 fps processing.
Common practice holds out the last 8 train videos (33-40) for validation.
"""
import csv
import os

import numpy as np

N_PHASES = 7
N_TOOLS = 7
TRAIN_IDS = [f"{i:02d}" for i in range(1, 41)]
TEST_IDS = [f"{i:02d}" for i in range(41, 81)]
VAL_IDS = [f"{i:02d}" for i in range(33, 41)]
SRC_FPS = 25
DST_FPS = 1


PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]
_PHASE_ID = {p.casefold(): i for i, p in enumerate(PHASE_NAMES)}


def _phase_to_id(token):
    token = token.strip()
    try:
        return int(token)
    except ValueError:
        pass
    key = token.replace(" ", "").replace("_", "").casefold()
    if key not in _PHASE_ID:
        raise ValueError(f"unknown phase {token!r}")
    return _PHASE_ID[key]


def parse_phase_txt(path):
    """Return int array of 0-based phase ids, one row per annotated frame (25 fps).

    Real files use a "Frame<TAB>Phase" header and phase *names*; integer ids
    are accepted too.
    """
    phases = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.replace(",", "\t").split("\t") if p != ""]
            try:
                phases.append(_phase_to_id(parts[-1]))
            except ValueError:
                if len(phases) == 0:
                    continue  # header line
                raise
    arr = np.asarray(phases, dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() >= N_PHASES):
        raise ValueError(f"{path}: phase ids out of range 0..{N_PHASES - 1}")
    return arr


def parse_tool_txt(path):
    """Return (N, 7) binary array. Robust to header row and frame-number column."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.replace(",", "\t").split("\t") if p != ""]
            try:
                nums = [int(float(p)) for p in parts]
            except ValueError:
                continue  # header
            if len(nums) >= N_TOOLS + 1:
                nums = nums[1:N_TOOLS + 1]  # drop frame-number column
            rows.append(nums[:N_TOOLS] + [0] * max(0, N_TOOLS - len(nums)))
    return np.asarray(rows, dtype=np.int64)


def to_1fps(arr25, src_fps=SRC_FPS):
    """Downsample a 25 fps label track to 1 fps (take first frame of each second)."""
    return arr25[::src_fps]


def video_ids(split="all"):
    if split == "train":
        return TRAIN_IDS
    if split == "val":
        return VAL_IDS
    if split == "train_fit":
        return [v for v in TRAIN_IDS if v not in VAL_IDS]
    if split == "test":
        return TEST_IDS
    return [f"{i:02d}" for i in range(1, 81)]


def build_frame_manifest(root, out_csv):
    """One row per 1fps frame: video,frame,phase,tools. Videos lacking annotations skip."""
    rows = 0
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "frame", "phase", "tools"])
        for vid in video_ids():
            pp = os.path.join(root, "phase_annotations", f"video{vid}-phase.txt")
            if not os.path.isfile(pp):
                continue
            ph = to_1fps(parse_phase_txt(pp))
            tp = os.path.join(root, "tool_annotations", f"video{vid}-tool.txt")
            tools = parse_tool_txt(tp) if os.path.isfile(tp) else None
            for i, p in enumerate(ph):
                t = ""
                if tools is not None and i < len(tools):
                    t = "".join(str(int(x)) for x in tools[i])
                w.writerow([vid, i, int(p), t])
                rows += 1
    return rows


def class_weights_from_manifest(manifest_csv):
    """Median-frequency balancing (TeCNO standard): w_c = median(freq) / freq_c."""
    counts = np.zeros(N_PHASES)
    with open(manifest_csv) as fh:
        for r in csv.DictReader(fh):
            counts[int(r["phase"])] += 1
    freq = counts / max(counts.sum(), 1)
    med = float(np.median(freq[freq > 0])) if (freq > 0).any() else 1.0
    w = np.where(freq > 0, med / np.maximum(freq, 1e-12), 0.0)
    return w.astype(np.float32)
