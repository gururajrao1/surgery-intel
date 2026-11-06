"""Stage A: frame-level multitask training (EndoNet-style) on real frames.

Usage (after fetching videos + extracting frames + building manifest):
  python -m surgery_intel.train --manifest data/manifest.csv --frames data/frames_1fps \\
      --videos 13,37,12,28 --val-videos 28 --epochs 3 --backbone tiny --img-size 128

Without --manifest it runs a synthetic smoke epoch (no data needed).
Checkpoints the best val-accuracy weights to --out.
"""
import argparse
import os

import numpy as np
import torch

from .cholec80 import N_PHASES, class_weights_from_manifest
from .dataset import PhaseFrameDataset, make_loader
from .model import EndoNetBaseline


def split_weights(manifest, videos):
    if manifest is None:
        return None
    import csv
    counts = np.zeros(N_PHASES)
    with open(manifest) as fh:
        for r in csv.DictReader(fh):
            if r["video"] in videos and int(r["phase"]) >= 0:
                counts[int(r["phase"])] += 1
    freq = counts / max(counts.sum(), 1)
    med = float(np.median(freq[freq > 0])) if (freq > 0).any() else 1.0
    return torch.tensor(np.where(freq > 0, med / np.maximum(freq, 1e-12), 0.0),
                        dtype=torch.float32)


def run_epoch(model, dl, opt, ce, bce, train, amp=False, scaler=None):
    model.train(train)
    tot, correct, total = 0.0, 0, 0
    use_cuda = next(model.parameters()).is_cuda
    for img, ph, tools in dl:
        with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu",
                                enabled=amp):
            pl, tl = model(img)
            loss = ce(pl, ph) + bce(tl, tools)
        if train:
            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
        tot += loss.item()
        correct += (pl.argmax(-1) == ph).sum().item()
        total += ph.numel()
    return tot / max(len(dl), 1), correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--frames", default="data/frames_1fps")
    ap.add_argument("--videos", default=None, help="comma-separated video ids for training")
    ap.add_argument("--val-videos", default=None, help="comma-separated video ids for validation")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--backbone", default="tiny", choices=["tiny", "mobilenet", "resnet18", "resnet50"])
    ap.add_argument("--pretrained", action="store_true", help="ImageNet init (recommended)")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="outputs/frame_best.pt")
    ap.add_argument("--num-workers", type=int, default=2,
                    help="DataLoader workers (JPEG decode + augment off the train loop)")
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision: fp16 on CUDA, bf16 on CPU")
    a = ap.parse_args()

    if a.manifest and os.path.isfile(a.manifest):
        tr_vids = a.videos.split(",") if a.videos else None
        va_vids = a.val_videos.split(",") if a.val_videos else []
        tr = make_loader(PhaseFrameDataset(a.frames, a.manifest, videos=tr_vids,
                                           img_size=a.img_size, augment=True),
                         batch_size=a.batch_size, shuffle=True,
                         num_workers=a.num_workers)
        va = make_loader(PhaseFrameDataset(a.frames, a.manifest, videos=va_vids,
                                           img_size=a.img_size),
                         batch_size=a.batch_size, num_workers=a.num_workers)
        weights = split_weights(a.manifest, tr_vids or [])
    else:
        tr = make_loader(PhaseFrameDataset(None, None), batch_size=8, shuffle=True,
                         num_workers=0)
        va = make_loader(PhaseFrameDataset(None, None), batch_size=8, num_workers=0)
        weights = None
    if len(tr.dataset) == 0:
        raise SystemExit("no training rows (check --manifest/--videos/frames)")
    print(f"train rows={len(tr.dataset)} val rows={len(va.dataset)}")

    model = EndoNetBaseline(backbone=a.backbone, pretrained=a.pretrained)
    scaler = None
    if a.amp and torch.cuda.is_available():
        scaler = torch.amp.GradScaler("cuda")
    ce = torch.nn.CrossEntropyLoss(weight=weights)
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best = -1.0
    for ep in range(a.epochs):
        tl, ta = run_epoch(model, tr, opt, ce, bce, True, amp=a.amp, scaler=scaler)
        msg = f"ep {ep}: train loss {tl:.4f} acc {ta:.3f}"
        if len(va.dataset):
            vl, va_acc = run_epoch(model, va, opt, ce, bce, False, amp=a.amp)
            msg += f" | val loss {vl:.4f} acc {va_acc:.3f}"
            if va_acc > best:
                best = va_acc
                os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
                torch.save({"backbone": a.backbone, "pretrained": a.pretrained,
                            "img_size": a.img_size, "state": model.state_dict()}, a.out)
        print(msg, flush=True)
    print(f"best val acc {best:.3f} -> {a.out}" if best >= 0 else "done (no val set)")


if __name__ == "__main__":
    main()
