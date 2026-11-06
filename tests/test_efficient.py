import time
import torch
from surgery_intel.temporal import CausalTCNHead, DualPathSSMHead, SelectiveSSM
from surgery_intel.efficient_model import (
    EfficientRecognizer, count_params, quantize_temporal, script_encoder,
)
from surgery_intel.model import EndoNetBaseline


def test_causal_no_future_leak():
    torch.manual_seed(0)
    head = CausalTCNHead(dim=16, layers=2, num_phases=3)
    head.eval()
    x = torch.randn(1, 8, 16)
    y1 = head(x)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(1, 3, 16)  # perturb future
    y2 = head(x2)
    # frames 0..4 must be unaffected by future change (causal)
    assert torch.allclose(y1[:, :5], y2[:, :5], atol=1e-5)


def test_ssm_streaming_matches_full():
    torch.manual_seed(1)
    ssm = SelectiveSSM(dim=16)
    ssm.eval()
    x = torch.randn(1, 6, 16)
    full, _ = ssm(x)
    state = torch.zeros(1, 16)
    outs = []
    for t in range(6):
        y, state = ssm.step(x[:, t], state)
        outs.append(y)
    streamed = torch.stack(outs, dim=1)
    assert torch.allclose(full, streamed, atol=1e-5)


def test_efficient_forward_and_params():
    m = EfficientRecognizer(backbone="tiny", feat_dim=64, head="ssm")
    vid = torch.rand(1, 4, 3, 64, 64)
    phase, tools = m(vid)
    assert phase.shape == (1, 4, 7) and tools.shape == (1, 4, 7)
    base = EndoNetBaseline()
    assert count_params(m) < count_params(base), \
        f"efficient {count_params(m)} should be < baseline {count_params(base)}"


def test_tcn_forward_shape():
    m = EfficientRecognizer(backbone="tiny", feat_dim=64, head="tcn")
    vid = torch.rand(1, 4, 3, 64, 64)
    phase, tools = m(vid)
    assert phase.shape == (1, 4, 7)


def test_multistage_tcn_and_smoothing():
    from surgery_intel.temporal import MultiStageTCN, smooth_loss
    m = MultiStageTCN(dim=32, layers=2, num_phases=7).eval()
    x = torch.rand(1, 8, 32)
    with torch.no_grad():
        stages = m(x)
    assert len(stages) == 2 and stages[-1].shape == (1, 8, 7)
    s = smooth_loss(stages[-1])
    assert s.item() >= 0
    # constant logits -> zero temporal variation -> ~zero smoothing
    assert smooth_loss(torch.zeros(1, 8, 7)).item() < 1e-6


def test_resnet50_backbone_option():
    import torchvision
    m = EfficientRecognizer(backbone="resnet50", feat_dim=64, head="ssm")
    assert m.encode(torch.rand(1, 3, 64, 64)).shape == (1, 64)


def test_temporal_dropout_eval_identity_and_stream():
    torch.manual_seed(0)
    m = EfficientRecognizer(backbone="tiny", feat_dim=64, head="ssm").eval()
    m.temporal.drop.p = 0.5  # eval mode: still identity
    vid = torch.rand(1, 4, 3, 64, 64)
    with torch.no_grad():
        phase, tools = m(vid)
        state = m.init_state(1, vid.device)
        outs = []
        for t in range(4):
            p, tl, state = m.stream_step(vid[:, t], state)
            outs.append(p)
    assert phase.shape == (1, 4, 7) and torch.isfinite(phase).all()
    assert torch.isfinite(torch.stack(outs)).all() and torch.isfinite(state).all()


def test_scripted_encoder_matches():
    m = EfficientRecognizer(backbone="tiny", feat_dim=64, head="ssm").eval()
    enc = script_encoder(m)
    x = torch.rand(4, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(enc(x), m.encode(x), atol=1e-5)


def test_quantized_temporal_close():
    m = EfficientRecognizer(backbone="tiny", feat_dim=64, head="ssm").eval()
    q = quantize_temporal(m)
    feats = torch.rand(1, 8, 64)
    with torch.no_grad():
        ref = m.temporal(feats)
        got = q.temporal(feats)
    assert torch.allclose(ref, got, atol=5e-2)
