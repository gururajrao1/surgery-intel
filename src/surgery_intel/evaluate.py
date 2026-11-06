"""Paper-style evaluation: per-video metrics + mean/std over the test split.

Reads preds.csv (video,frame,phase) and the frame manifest; tolerance is given
in seconds (papers use ~10 s relaxed boundaries).
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

from .cholec80 import N_PHASES, video_ids
from .metrics import video_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--relaxed-sec", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=1.0)
    a = ap.parse_args()

    gt = defaultdict(dict)
    with open(a.manifest) as fh:
        for r in csv.DictReader(fh):
            gt[r["video"]][int(r["frame"])] = int(r["phase"])
    pred = defaultdict(dict)
    with open(a.preds) as fh:
        for r in csv.DictReader(fh):
            pred[r["video"]][int(r["frame"])] = int(r["phase"])

    keys = ["accuracy", "precision", "recall", "jaccard", "f1"]
    agg = {}
    for tag, sec in [("strict", 0.0), ("relaxed", a.relaxed_sec)]:
        tol = int(round(sec * a.fps))
        per_video = {}
        for vid in video_ids("test"):
            if vid not in gt or vid not in pred:
                continue
            frames = sorted(set(gt[vid]) & set(pred[vid]))
            if not frames:
                continue
            g = np.array([gt[vid][f] for f in frames])
            p = np.array([pred[vid][f] for f in frames])
            per_video[vid] = video_scores(g, p, num_phases=N_PHASES, tol_frames=tol)
        agg[tag] = per_video
        print(f"== {tag} (tol {sec}s): {len(per_video)} videos ==")
        print(f"{'video':>8} " + " ".join(f"{k:>9}" for k in keys))
        for vid in sorted(per_video):
            s = per_video[vid]
            print(f"{vid:>8} " + " ".join(f"{s[k] * 100:9.2f}" for k in keys))
        if per_video:
            arr = {k: np.array([s[k] for s in per_video.values()]) for k in keys}
            print("mean    " + " ".join(f"{arr[k].mean() * 100:9.2f}" for k in keys))
            print("std     " + " ".join(f"{arr[k].std() * 100:9.2f}" for k in keys))


if __name__ == "__main__":
    main()
