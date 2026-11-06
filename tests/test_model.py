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
