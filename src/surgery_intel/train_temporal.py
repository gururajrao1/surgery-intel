"""Stage B training: temporal head on cached per-video frame features.

Usage:
  python scripts/cache_features.py --data data/cholec80 --frames data/frames_1fps --out data/features
  python -m surgery_intel.train_temporal --features data/features --epochs 20
"""
import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .cholec80 import N_PHASES, class_weights_from_manifest, video_ids
from .temporal import CausalTCNHead, DualPathSSMHead, MultiStageTCN, smooth_loss


class FeatureClipDataset(Dataset):
    def __init__(self, feat_dir, manifest_csv, split, clip_len=256):
        self.items = []
        with open(manifest_csv) as fh:
            by_video = {}
            for r in csv.DictReader(fh):
                by_video.setdefault(r["video"], []).append(r)
        for vid in video_ids(split):
            rows = sorted(by_video.get(vid, []), key=lambda r: int(r["frame"]))
            fp = os.path.join(feat_dir, f"{vid}.npy")
            if rows and os.path.isfile(fp):
                self.items.append((fp, np.array([int(r["phase"]) for r in rows])))
        self.clip_len = clip_len

    def __len__(self):
        return max(len(self.items) * 4, 1)

    def __getitem__(self, i):
        fp, ph = self.items[i % len(self.items)]
        feats = np.load(fp).astype(np.float32)
        n = len(feats)
        if n >= self.clip_len:
            s = np.random.randint(0, n - self.clip_len + 1)
            feats, ph = feats[s:s + self.clip_len], ph[s:s + self.clip_len]
        return torch.from_numpy(feats), torch.from_numpy(ph).long()


def make_head(feat_dim, head):
    if head == "tcn":
        return CausalTCNHead(feat_dim, num_phases=N_PHASES)
    if head == "mtcn":
        return MultiStageTCN(feat_dim, num_phases=N_PHASES)
    return DualPathSSMHead(feat_dim, num_phases=N_PHASES)


def run_epoch(head, dl, opt, weights, train, smooth_lam=0.15, smooth_tau=4.0):
    head.train(train)
    tot, correct, total = 0.0, 0, 0
    ce = torch.nn.CrossEntropyLoss(weight=weights)
    for feats, ph in dl:
        out = head(feats)
        stages = out if isinstance(out, list) else [out]
        loss = 0.0
        for logits in stages:
            loss = loss + ce(logits.reshape(-1, N_PHASES), ph.reshape(-1))
            loss = loss + smooth_loss(logits, tau=smooth_tau, lam=smooth_lam)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot += loss.item()
        pred = stages[-1].argmax(-1)
        correct += (pred == ph).sum().item()
        total += ph.numel()
    return tot / max(len(dl), 1), correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--head", default="ssm", choices=["ssm", "tcn", "mtcn"])
    ap.add_argument("--smooth-lam", type=float, default=0.15)
    ap.add_argument("--smooth-tau", type=float, default=4.0)
    ap.add_argument("--feat-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--clip-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="outputs/temporal_best.pt")
    a = ap.parse_args()

    from torch.utils.data import DataLoader
    weights = None
    if os.path.isfile(a.manifest):
        weights = torch.tensor(class_weights_from_manifest(a.manifest))
    tr = DataLoader(FeatureClipDataset(a.features, a.manifest, "train_fit", a.clip_len),
                    batch_size=2, shuffle=True)
    va = DataLoader(FeatureClipDataset(a.features, a.manifest, "val", a.clip_len),
                    batch_size=2)
    head = make_head(a.feat_dim, a.head)
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    best = -1.0
    no_val = len(va.dataset) == 0
    for ep in range(a.epochs):
        tl, ta = run_epoch(head, tr, opt, weights, True, a.smooth_lam, a.smooth_tau)
        vl, va_acc = (tl, ta) if no_val else run_epoch(
            head, va, opt, weights, False, a.smooth_lam, a.smooth_tau)
        print(f"ep {ep}: train loss {tl:.4f} acc {ta:.3f} | val loss {vl:.4f} acc {va_acc:.3f}", flush=True)
        if va_acc > best:
            best = va_acc
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            torch.save({"head": a.head, "feat_dim": a.feat_dim,
                        "state": head.state_dict()}, a.out)
    print(f"best val acc {best:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
