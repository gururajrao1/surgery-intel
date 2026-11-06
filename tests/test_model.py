import torch
from surgery_intel.model import EndoNetBaseline, multitask_loss


def test_forward_shapes():
    m = EndoNetBaseline()
    x = torch.rand(2, 3, 250, 250)
    pl, tl = m(x)
    assert pl.shape == (2, 7)
    assert tl.shape == (2, 7)


def test_backbone_option_selects_encoder():
    from surgery_intel.efficient_model import count_params
    tiny = EndoNetBaseline(backbone="tiny")
    base = EndoNetBaseline(backbone="resnet18")
    assert count_params(tiny) < count_params(base) // 10
    x = torch.rand(1, 3, 64, 64)
    assert tiny(x)[0].shape == (1, 7)
    assert base(x)[0].shape == (1, 7)


def test_loss_runs():
    m = EndoNetBaseline()
    x = torch.rand(2, 3, 250, 250)
    pl, tl = m(x)
    loss, ce, bce = multitask_loss(pl, tl, torch.tensor([0, 1]), torch.randint(0, 2, (2, 7)))
    assert loss.detach().item() > 0


def test_dropout_eval_identity():
    torch.manual_seed(0)
    ref = EndoNetBaseline(backbone="tiny", dropout=0.0).eval()
    torch.manual_seed(0)
    m = EndoNetBaseline(backbone="tiny", dropout=0.5).eval()
    m.load_state_dict(ref.state_dict(), strict=False)
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(ref(x)[0], m(x)[0])


def test_freeze_encoder():
    m = EndoNetBaseline(backbone="tiny")
    m.freeze_encoder()
    assert all(not p.requires_grad for p in m.backbone.parameters())
    assert any(p.requires_grad for p in m.fc7.parameters())
    m.train()
    assert not m.backbone.training  # BN stats stay fixed while heads train
    m.unfreeze_encoder()
    assert all(p.requires_grad for p in m.backbone.parameters())
