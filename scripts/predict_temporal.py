"""Temporal-head prediction: cached features + trained head -> preds csv.

Usage: python scripts/predict_temporal.py --videos 70 --features data/features10
       --ckpt outputs/temporal_ssm_pre10_v33.pt --out outputs/p70_t10.csv
"""
import argparse
import csv
import os

import numpy as np
import torch

from surgery_intel.train_temporal import make_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, "src")
    from surgery_intel.train import resolve_device
    device = resolve_device(a.device)
    print(f"device: {device}")
    ckpt = torch.load(a.ckpt, map_location="cpu")
    head = make_head(ckpt.get("feat_dim", 512), ckpt.get("head", "ssm")).eval().to(device)
    head.load_state_dict(ckpt["state"])
    gt = {}
    with open(a.manifest) as fh:
        for r in csv.DictReader(fh):
            gt.setdefault(r["video"], []).append(r)
    rows = []
    with torch.no_grad():
        for vid in a.videos:
            fp = os.path.join(a.features, f"{vid}.npy")
            feats = torch.from_numpy(np.load(fp).astype(np.float32)).unsqueeze(0).to(device)
            out = head(feats)
            logits = out[-1] if isinstance(out, list) else out
            frames = sorted(gt.get(vid, []), key=lambda r: int(r["frame"]))
            for r, p in zip(frames, logits[0].argmax(-1).tolist()):
                rows.append({"video": vid, "frame": r["frame"], "phase": p})
            print(f"{vid}: {len(frames)} preds")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "frame", "phase"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
