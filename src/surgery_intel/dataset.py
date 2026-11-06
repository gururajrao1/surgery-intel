"""Cholec80 manifest + frame dataset (1fps, 40/40 split)."""
import argparse
import csv
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


def video_ids(lo, hi):
    return [f"{i:02d}" for i in range(lo, hi + 1)]


def make_loader(ds, batch_size, shuffle=False, num_workers=2):
    """DataLoader with sane throughput defaults.

    Workers move JPEG decode + augment off the training process; pin_memory +
    persistent workers only apply when workers are actually used. All entry
    points calling this run under ``if __name__ == "__main__"``, so worker
    spawn is safe on Windows too.
    """
    kw = {}
    if num_workers and num_workers > 0:
        kw = {"num_workers": num_workers,
              "pin_memory": torch.cuda.is_available(),  # warns + no-ops on CPU-only
              "persistent_workers": True, "prefetch_factor": 2}
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kw)


def build_manifest(frames_root, out_csv):
    rows = []
    if os.path.isdir(frames_root):
        for vid in sorted(os.listdir(frames_root)):
            d = os.path.join(frames_root, vid)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.endswith(".jpg"):
                    rows.append({"video": vid, "frame": f, "phase": -1, "tools": ""})
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "frame", "phase", "tools"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


class PhaseFrameDataset(Dataset):
    """Frame rows from a manifest. frame may be an int index (-> %08d.jpg) or a filename."""

    def __init__(self, frames_root, manifest_csv=None, videos=None, img_size=250, augment=False):
        self.frames_root = frames_root or os.path.join("data", "frames_1fps")
        tf_list = [T.Resize((img_size, img_size))]
        if augment:
            # Two-stage spirit: geometric offline-style + photometric/blur online,
            # so the model rarely sees the same frame twice (cf. DIST/MomentNet).
            tf_list += [T.RandomHorizontalFlip(),
                        T.RandomRotation(10),
                        T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))]
        tf_list.append(T.ToTensor())
        self.tf = T.Compose(tf_list)
        self.items = []
        if manifest_csv and os.path.isfile(manifest_csv):
            import csv as _csv
            with open(manifest_csv) as fh:
                for r in _csv.DictReader(fh):
                    if videos is not None and r["video"] not in videos:
                        continue
                    if int(r["phase"]) < 0:
                        continue
                    self.items.append(r)
        else:
            # synthetic fallback so smoke tests run without Cholec80
            rng = torch.Generator().manual_seed(0)
            for i in range(64):
                self.items.append({"synthetic": True, "phase": int(torch.randint(0, 7, (1,), generator=rng))})

    def __len__(self):
        return len(self.items)

    def _tools(self, s):
        s = (s or "").strip()
        if len(s) == 7 and set(s) <= {"0", "1"}:
            return torch.tensor([float(c) for c in s])
        return torch.zeros(7)

    def __getitem__(self, i):
        it = self.items[i]
        if "synthetic" in it:
            g = torch.Generator().manual_seed(i)
            img = torch.rand(3, 250, 250, generator=g)
            return img, torch.tensor(it["phase"]), torch.randint(0, 2, (7,), generator=g).float()
        f = it["frame"]
        try:
            fname = f"{int(f):08d}.jpg"
        except (TypeError, ValueError):
            fname = str(f)
        p = os.path.join(self.frames_root, it["video"], fname)
        img = self.tf(Image.open(p).convert("RGB"))
        return img, torch.tensor(int(it["phase"])), self._tools(it.get("tools"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-manifest", action="store_true")
    ap.add_argument("--frames", default="data/frames_1fps")
    ap.add_argument("--out", default="data/manifest.csv")
    a = ap.parse_args()
    if a.build_manifest:
        n = build_manifest(a.frames, a.out)
        print(f"wrote {n} rows to {a.out}")
