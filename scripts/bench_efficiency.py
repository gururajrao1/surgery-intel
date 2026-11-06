"""Benchmark: baseline vs efficient variants — params + per-frame latency (synthetic CPU)."""
import argparse
import time
import torch
from surgery_intel.efficient_model import (
    EfficientRecognizer, build_backbone, count_params, quantize_temporal, script_encoder,
)
from surgery_intel.model import EndoNetBaseline


def time_fn(fn, iters=10):
    with torch.no_grad():
        for _ in range(3):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--frames", type=int, default=32)
    a = ap.parse_args()
    S = a.img_size
    L = a.frames
    torch.manual_seed(0)
    print(f"== {S}x{S}, {L}-frame clip, CPU ==")
    base = EndoNetBaseline().eval()
    tcn = EfficientRecognizer(backbone="tiny", feat_dim=128, head="tcn").eval()
    ssm = EfficientRecognizer(backbone="tiny", feat_dim=128, head="ssm").eval()
    frames = torch.rand(L, 3, S, S)
    vid = torch.rand(1, L, 3, S, S)

    ms_base = time_fn(lambda: base(frames)) / L * 1000
    print(f"EndoNetBaseline/resnet18-frame: params={count_params(base):,} ms/frame~{ms_base:.2f}")
    for name, m in [("Efficient/tiny+TCN", tcn), ("Efficient/tiny+DualSSM", ssm)]:
        ms = time_fn(lambda: m(vid)) / L * 1000
        print(f"{name}: params={count_params(m):,} ms/frame~{ms:.2f}")

    # Deployment variants of the SSM model
    try:
        enc = script_encoder(ssm)
        ms = time_fn(lambda: enc(frames)) / L * 1000
        print(f"SSM encoder TorchScripted: ms/frame~{ms:.2f} (frame part only)")
    except Exception as e:
        print(f"TorchScript export failed: {e}")
    try:
        q = quantize_temporal(ssm)
        feats = ssm.encode(vid)
        ms = time_fn(lambda: q.temporal(feats)) / L * 1000
        print(f"SSM temporal int8-dynamic: ms/frame~{ms:.2f} (temporal part only)")
    except Exception as e:
        print(f"int8 quantization failed: {e}")

    # Backbone encoder-only comparison (the accuracy lever: bigger backbone, same head)
    print("-- encoder-only ms/frame --")
    for bname in ["tiny", "mobilenet", "efficientnet_b0", "convnext_tiny", "resnet18"]:
        try:
            enc, _ = build_backbone(bname)
            enc.eval()
            ms = time_fn(lambda: enc(frames)) / L * 1000
            n = sum(p.numel() for p in enc.parameters())
            print(f"encoder/{bname}: params={n:,} ms/frame~{ms:.2f}")
        except Exception as e:
            print(f"encoder/{bname}: unavailable ({e})")
    print("SSM stream state: (1, 128) O(d) constant vs attention O(L*d)")


if __name__ == "__main__":
    main()
