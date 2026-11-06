"""Eval: compute phase metrics from preds vs gt CSVs."""
import argparse
import csv
import numpy as np
from .metrics import video_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--relaxed-sec", type=float, default=0.0)
    ap.add_argument("--fps", type=float, default=1.0)
    a = ap.parse_args()
    with open(a.gt) as fh:
        gt = [int(r["phase"]) for r in csv.DictReader(fh) if int(r["phase"]) >= 0]
    with open(a.preds) as fh:
        pred = [int(r["phase"]) for r in csv.DictReader(fh)]
    n = min(len(gt), len(pred))
    tol = int(round(a.relaxed_sec * a.fps))
    print(video_scores(np.array(gt[:n]), np.array(pred[:n]), tol_frames=tol))


if __name__ == "__main__":
    main()
