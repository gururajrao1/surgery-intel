"""Frame-level prediction: EndoNetBaseline ckpt -> preds csv (video,frame,phase).

Usage: python scripts/predict_frame.py --videos 70 --ckpt outputs/frame_mob_pre10_v33.pt --out outputs/p70_f10.csv
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
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision inference: fp16 on CUDA, bf16 on CPU")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ckpt = torch.load(a.ckpt, map_location="cpu")
    model = EndoNetBaseline(backbone=ckpt.get("backbone", "mobilenet")).eval()
    model.load_state_dict(ckpt["state"])
    use_cuda = next(model.parameters()).is_cuda
    rows = []
    with torch.no_grad():
        for vid in a.videos:
            ds = PhaseFrameDataset(a.frames, a.manifest, videos=[vid], img_size=a.img_size)
            dl = make_loader(ds, batch_size=a.batch_size, num_workers=a.num_workers)
            k = 0
            for img, _, _ in dl:
                with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu",
                                        enabled=a.amp):
                    logits, _ = model(img)
                for p in logits.argmax(-1).tolist():
                    rows.append({"video": vid, "frame": ds.items[k]["frame"], "phase": p})
                    k += 1
            print(f"{vid}: {k} preds")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "frame", "phase"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
