import csv
import os

import numpy as np
import torch

from surgery_intel.cholec80 import (
    build_frame_manifest, class_weights_from_manifest, parse_phase_txt,
    parse_tool_txt, to_1fps, video_ids,
)


def _write_tree(root):
    pa = os.path.join(root, "phase_annotations")
    ta = os.path.join(root, "tool_annotations")
    os.makedirs(pa)
    os.makedirs(ta)
    # video01: 50 rows @25fps -> 2 rows @1fps, phases 3 then 5
    with open(os.path.join(pa, "video01-phase.txt"), "w") as fh:
        for i in range(50):
            fh.write(f"{i}\t{3 if i < 25 else 5}\n")
    with open(os.path.join(ta, "video01-tool.txt"), "w") as fh:
        fh.write("Frame\tG\tB\tH\tS\tC\tI\tSB\n")
        for i in range(2):
            fh.write(f"{i}\t1\t0\t0\t0\t0\t0\t0\n")
    # video41 (test split): 25 rows -> 1 row @1fps, phase 0
    with open(os.path.join(pa, "video41-phase.txt"), "w") as fh:
        for i in range(25):
            fh.write(f"{i}\t0\n")


def test_parse_and_downsample(tmp_path):
    _write_tree(str(tmp_path))
    ph = parse_phase_txt(os.path.join(str(tmp_path), "phase_annotations", "video01-phase.txt"))
    assert len(ph) == 50 and ph[0] == 3 and ph[-1] == 5
    assert list(to_1fps(ph)) == [3, 5]
    tools = parse_tool_txt(os.path.join(str(tmp_path), "tool_annotations", "video01-tool.txt"))
    assert tools.shape == (2, 7) and tools[0, 0] == 1 and tools.sum() == 2


def test_phase_names_real_format(tmp_path):
    p = os.path.join(str(tmp_path), "names-phase.txt")
    with open(p, "w") as fh:
        fh.write("Frame\tPhase\n0\tPreparation\n1\tCalotTriangleDissection\n2\t0\n")
    arr = parse_phase_txt(p)
    assert list(arr) == [0, 1, 0]


def test_manifest_splits_weights(tmp_path):
    _write_tree(str(tmp_path))
    out = os.path.join(str(tmp_path), "manifest.csv")
    n = build_frame_manifest(str(tmp_path), out)
    assert n == 3  # 2 train rows + 1 test row
    assert video_ids("train")[0] == "01" and video_ids("test")[0] == "41"
    assert "33" in video_ids("val") and "33" not in video_ids("train_fit")
    w = class_weights_from_manifest(out)
    assert len(w) == 7 and (w >= 0).all()


def test_median_frequency_balancing(tmp_path):
    man = os.path.join(str(tmp_path), "manifest.csv")
    with open(man, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "frame", "phase", "tools"])
        for i in range(10):
            w.writerow(["01", i, 0, ""])  # majority
        for i in range(2):
            w.writerow(["01", 10 + i, 1, ""])  # minority
    w = class_weights_from_manifest(man)
    assert w[0] < w[1]  # minority up-weighted
    assert abs(w[1] - (6 / 12) / (2 / 12)) < 1e-5  # median freq / class freq


def test_temporal_train_smoke(tmp_path):
    from torch.utils.data import DataLoader
    from surgery_intel.train_temporal import FeatureClipDataset, make_head, run_epoch
    man = os.path.join(str(tmp_path), "manifest.csv")
    with open(man, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "frame", "phase", "tools"])
        for i in range(8):
            w.writerow(["01", i, i % 2, ""])
    os.makedirs(tmp_path / "feat", exist_ok=True)
    np.save(os.path.join(str(tmp_path), "feat", "01.npy"), np.random.rand(8, 16).astype(np.float32))
    ds = FeatureClipDataset(os.path.join(str(tmp_path), "feat"), man, "train_fit", clip_len=4)
    dl = DataLoader(ds, batch_size=2)
    head = make_head(16, "ssm")
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss, acc = run_epoch(head, dl, opt, None, True)
    assert loss > 0 and 0.0 <= acc <= 1.0
