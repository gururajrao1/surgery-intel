"""Efficient causal temporal heads: TCN + selective-SSM (Mamba-style), O(L) streaming.

Research basis:
- TeCNO: dilated causal TCN, O(L*K) vs Transformer O(L^2).
- SR-Mamba / SurgicalMamba / HID-SSM: structured state-space duality, per-frame O(d),
  92.6-96.1% on Cholec80 with fewer params / ~2G FLOPs saved vs attention.
This file implements dependency-free PyTorch versions with identical complexity behavior.
"""
import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, c, kernel=3, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c, c, kernel, padding=0, dilation=dilation, groups=c)

    def forward(self, x):
        # x: (B, C, L); left-pad only -> causal
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, dim, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        self.dw = CausalConv1d(dim, kernel, dilation)
        self.pw = nn.Conv1d(dim, dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, D)
        h = x.transpose(1, 2)
        h = self.dw(h)
        h = self.pw(h)
        h = h.transpose(1, 2)
        h = self.drop(h)
        out = self.norm(x + h)
        return torch.relu(out)


class CausalTCNHead(nn.Module):
    """Stacked dilated causal TCN. Receptive field = 1 + 2*sum(dilations). Linear in L."""
    def __init__(self, dim=256, layers=4, kernel=3, num_phases=7):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList(
            [TCNBlock(dim, kernel, dilation=2 ** i) for i in range(layers)]
        )
        self.head = nn.Linear(dim, num_phases)

    def forward(self, feats):
        # feats: (B, L, D)
        h = torch.relu(self.proj_in(feats))
        for b in self.blocks:
            h = b(h)
        return self.head(h)


class SelectiveSSM(nn.Module):
    """Minimal data-dependent linear recurrence mimicking Mamba core complexity.

    h_t = decay_t * h_{t-1} + inp_t * x_t,  decay/selective gates from x_t.
    Per-frame O(d), total O(L*d), streaming with fixed state (B, D).
    """
    def __init__(self, dim):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim * 2)
        self.decay_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feats, state=None):
        # feats: (B, L, D) -> (B, L, D); state: (B, D) or None
        B, L, D = feats.shape
        h = torch.zeros(B, D, device=feats.device) if state is None else state
        outs = []
        for t in range(L):
            x = feats[:, t]
            xb, gate = self.in_proj(x).chunk(2, dim=-1)
            decay = torch.sigmoid(self.decay_proj(x))  # (B, D) in (0,1)
            h = decay * h + (1 - decay) * xb
            y = self.norm(h) * torch.sigmoid(gate)
            outs.append(self.out_proj(y))
        return torch.stack(outs, dim=1), h

    @torch.no_grad()
    def step(self, x, state):
        """Streaming single frame: x (B,D), state (B,D) -> (y (B,D), new_state)."""
        xb, gate = self.in_proj(x).chunk(2, dim=-1)
        decay = torch.sigmoid(self.decay_proj(x))
        h = decay * state + (1 - decay) * xb
        y = self.out_proj(self.norm(h) * torch.sigmoid(gate))
        return y, h


def smooth_loss(logits, tau=4.0, lam=0.15):
    """MS-TCN truncated-MSE smoothing over frame-wise log-probs (anti-over-segmentation)."""
    logp = logits.log_softmax(dim=-1)
    diff = (logp[:, 1:] - logp[:, :-1]).pow(2).clamp(max=tau).mean()
    return lam * diff


class MultiStageTCN(nn.Module):
    """Two-stage MS-TCN refinement (TeCNO lineage): stage 2 refines stage-1 probs."""
    def __init__(self, dim=256, layers=4, num_phases=7):
        super().__init__()
        self.stage1 = CausalTCNHead(dim, layers=layers, num_phases=num_phases)
        self.stage2 = CausalTCNHead(num_phases, layers=layers, num_phases=num_phases)

    def forward(self, feats):
        l1 = self.stage1(feats)
        l2 = self.stage2(l1.softmax(dim=-1))
        return [l1, l2]


class DualPathSSMHead(nn.Module):
    """SurgicalMamba-style dual path: short-term depthwise conv + long-term SSM, gated fuse.

    Short path captures instrument motion bursts; long path holds phase context.
    Both paths causal + streaming-capable, total O(L*d).
    """
    def __init__(self, dim=256, num_phases=7):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.short = CausalConv1d(dim, kernel=3, dilation=1)
        self.long = SelectiveSSM(dim)
        self.fuse = nn.Linear(dim * 2, dim)
        self.head = nn.Linear(dim, num_phases)

    def forward(self, feats):
        h = torch.relu(self.proj(feats))  # (B,L,D)
        s = self.short(h.transpose(1, 2)).transpose(1, 2)  # short-term
        l, _ = self.long(h)  # long-term
        fused = torch.relu(self.fuse(torch.cat([s, l], dim=-1)))
        return self.head(fused)
