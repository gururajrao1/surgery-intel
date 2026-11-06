"""Online inference for one video's frame folder -> preds rows (video,frame,phase).

Streams frame-by-frame through EfficientRecognizer (SSM keeps O(d) state),
so memory stays flat on hour-long videos. Output matches viewer.html format.
"""
import argparse
import csv
import os

import torch
from PIL import Image
import torchvision.transforms as T

from .efficient_model import EfficientRecognizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="folder of 00000000.jpg frames (1 fps)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default=None, help="temporal .pt from train_temporal (optional)")
    ap.add_argument("--backbone", default="tiny")
    ap.add_argument("--feat-dim", type=int, default=256)
    ap.add_argument("--out", default="outputs/preds.csv")
    a = ap.parse_args()

    tf = T.Compose([T.Resize((250, 250)), T.ToTensor()])
    model = EfficientRecognizer(backbone=a.backbone, feat_dim=a.feat_dim, head="ssm").eval()
    if a.weights and os.path.isfile(a.weights):
        ckpt = torch.load(a.weights, map_location="cpu")
        model.temporal.load_state_dict(ckpt["state"])
    files = sorted(f for f in os.listdir(a.frames) if f.endswith(".jpg"))
    state = model.init_state(1, "cpu")
    rows = []
    with torch.no_grad():
        for i, f in enumerate(files):
            img = tf(Image.open(os.path.join(a.frames, f)).convert("RGB")).unsqueeze(0)
            phase, _, state = model.stream_step(img, state)
            rows.append((a.video, i, int(phase.argmax(-1).item())))
    new = not os.path.isfile(a.out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["video", "frame", "phase"])
        w.writerows(rows)
    print(f"wrote {len(rows)} preds for video {a.video} -> {a.out}")


if __name__ == "__main__":
    main()
