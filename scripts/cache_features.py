"""Cache per-frame fc7 features for the temporal stage (stage A -> stage B).

Usage: python scripts/cache_features.py --videos 13 --ckpt outputs/frame_tiny13.pt
Writes data/features/<vid>.npy (float32, one row per manifest frame).
"""
import argparse
import csv
import os

import torch

from surgery_intel.dataset import PhaseFrameDataset, make_loader
from surgery_intel.model import EndoNetBaseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--frames", default="data/frames_1fps")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision encode: fp16 on CUDA, bf16 on CPU")
    a = ap.parse_args()

    ckpt = torch.load(a.ckpt, map_location="cpu")
    model = EndoNetBaseline(backbone=ckpt.get("backbone", "resnet18")).eval()
    model.load_state_dict(ckpt["state"])
    feats_mod = torch.nn.Sequential(model.backbone, torch.nn.Flatten(1),
                                    model.fc7, torch.nn.ReLU())
    use_cuda = next(model.parameters()).is_cuda
    os.makedirs(a.out, exist_ok=True)
    with torch.no_grad():
        for vid in a.videos:
            ds = PhaseFrameDataset(a.frames, a.manifest, videos=[vid], img_size=128)
            if len(ds) == 0:
                print(f"{vid}: no rows, skipped")
                continue
            dl = make_loader(ds, batch_size=a.batch_size, num_workers=a.num_workers)
            with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu",
                                    enabled=a.amp):
                feats = torch.cat([feats_mod(img) for img, _, _ in dl]).numpy()
            np_path = os.path.join(a.out, f"{vid}.npy")
            __import__("numpy").save(np_path, feats.astype("float32"))
            print(f"{vid}: {feats.shape} -> {np_path}")


if __name__ == "__main__":
    main()
