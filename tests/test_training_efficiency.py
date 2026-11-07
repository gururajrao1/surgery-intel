"""Efficiency options: cached feature loads, loader flags, AMP run paths."""
import numpy as np
import pytest
import torch

from surgery_intel.dataset import PhaseFrameDataset, make_loader
from surgery_intel.model import EndoNetBaseline
from surgery_intel.train import run_epoch as frame_run_epoch
from surgery_intel.train_temporal import FeatureClipDataset
from surgery_intel.train_temporal import run_epoch as temporal_run_epoch


def _manifest(tmp_path):
    p = tmp_path / "manifest.csv"
    rows = ["video,frame,phase,tools"]
    for v in ("01", "02"):
        for f in range(300):
            rows.append(f"{v},{f},{f % 7},0000000")
    p.write_text("\n".join(rows) + "\n")
    return str(p)


def test_feature_dataset_loads_each_file_once(tmp_path, monkeypatch):
    feat_dir = tmp_path / "feats"
    feat_dir.mkdir()
    for v in ("01", "02"):
        np.save(feat_dir / f"{v}.npy", np.random.rand(300, 8).astype(np.float32))
    manifest = _manifest(tmp_path)
    # train_fit split = videos 01..32 minus val 33..40, so our 01/02 rows qualify
    import surgery_intel.train_temporal as tt

    calls = []
    real_load = np.load

    def counting_load(path, *a, **k):
        calls.append(str(path))
        return real_load(path, *a, **k)

    monkeypatch.setattr(np, "load", counting_load)
    ds = tt.FeatureClipDataset(str(feat_dir), manifest, "train_fit", clip_len=64)
    assert len(ds.items) == 2
    for i in range(len(ds)):
        feats, ph = ds[i]
        assert feats.shape == (64, 8) and ph.shape == (64,)
    # one load per video file total, not one per __getitem__
    assert sorted(calls) == sorted(str(feat_dir / f"{v}.npy") for v in ("01", "02"))


def test_make_loader_flags():
    ds = PhaseFrameDataset(None, None)
    ld = make_loader(ds, batch_size=8, shuffle=True, num_workers=2)
    assert ld.num_workers == 2
    assert ld.pin_memory is torch.cuda.is_available()
    ld0 = make_loader(ds, batch_size=8, num_workers=0)
    assert ld0.num_workers == 0
    batch = next(iter(ld0))
    assert batch[0].shape[1:] == (3, 250, 250)


def test_augmented_dataset_yields_valid_batch():
    ds = PhaseFrameDataset(None, None)  # synthetic rows (augment flag N/A offline)
    assert len(ds) == 64
    img, ph, tools = ds[0]
    assert img.shape == (3, 250, 250) and img.isfinite().all()
    # real-path augment transform chain builds and runs on a fake frame
    ds_aug = PhaseFrameDataset("nonexistent", None, img_size=128, augment=True)
    from PIL import Image
    x = ds_aug.tf(Image.new("RGB", (200, 200), (128, 64, 32)))
    assert x.shape == (3, 128, 128) and x.isfinite().all()


def test_build_opt_groups_and_freeze():
    from surgery_intel.train import build_opt, warmup_factor
    m = EndoNetBaseline(backbone="tiny")
    opt = build_opt(m, 1e-3, backbone_lr_mult=0.1)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert opt.param_groups[1]["lr"] == pytest.approx(1e-3)
    m.freeze_encoder()
    opt = build_opt(m, 1e-3, backbone_lr_mult=0.1)
    assert len(opt.param_groups) == 1 and opt.param_groups[0]["lr"] == 1e-3
    assert warmup_factor(0, 2) == 0.5 and warmup_factor(1, 2) == 1.0
    assert warmup_factor(5, 2) == 1.0 and warmup_factor(0, 0) == 1.0


def test_frame_run_epoch_amp_runs():
    torch.manual_seed(0)
    model = EndoNetBaseline(backbone="tiny")
    dl = make_loader(PhaseFrameDataset(None, None), batch_size=8, num_workers=0)
    ce = torch.nn.CrossEntropyLoss()
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, acc = frame_run_epoch(model, dl, opt, ce, bce, True, amp=True)
    assert np.isfinite(loss) and 0.0 <= acc <= 1.0


def test_temporal_run_epoch_amp_runs(tmp_path):
    torch.manual_seed(0)
    feat_dir = tmp_path / "feats"
    feat_dir.mkdir()
    for v in ("01", "02"):
        np.save(feat_dir / f"{v}.npy", np.random.rand(300, 16).astype(np.float32))
    manifest = _manifest(tmp_path)
    from surgery_intel.train_temporal import make_head

    ds = FeatureClipDataset(str(feat_dir), manifest, "train_fit", clip_len=64)
    dl = make_loader(ds, batch_size=2, num_workers=0)
    head = make_head(16, "ssm")
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss, acc = temporal_run_epoch(head, dl, opt, None, True, amp=True)
    assert np.isfinite(loss) and 0.0 <= acc <= 1.0
