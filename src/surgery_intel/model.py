"""EndoNet-style multitask model: backbone -> fc7 + fc_tool -> fc8 -> fc_phase."""
import torch
import torch.nn as nn
try:
    import torchvision.models as models
    _HAS_TV = True
except ImportError:
    models = None
    _HAS_TV = False


PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

TOOLS = [f"tool_{i}" for i in range(7)]


class EndoNetBaseline(nn.Module):
    def __init__(self, num_phases=7, num_tools=7, backbone="resnet18",
                 pretrained=False, feature_dim=512, hidden_dim=256,
                 dropout=0.0):
        super().__init__()
        self.backbone_name = backbone
        self._frozen_encoder = False
        self.drop = nn.Dropout(dropout)
        if backbone == "resnet18" and not _HAS_TV:
            # Lightweight fallback when torchvision is unavailable (tests/smoke).
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 7, stride=4, padding=3), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            in_dim = 32
        else:
            from .efficient_model import build_backbone
            self.backbone, in_dim = build_backbone(backbone, pretrained=pretrained)
        self.fc7 = nn.Linear(in_dim, feature_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc_tool = nn.Linear(feature_dim, num_tools)
        self.fc8 = nn.Linear(feature_dim + num_tools, hidden_dim)
        self.fc_phase = nn.Linear(hidden_dim, num_phases)

    def freeze_encoder(self):
        """Freeze backbone weights (fine-tune heads only); BN stays in eval."""
        self._frozen_encoder = True
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def unfreeze_encoder(self):
        self._frozen_encoder = False
        for p in self.backbone.parameters():
            p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if mode and self._frozen_encoder:
            # keep frozen BN stats fixed while heads still train
            self.backbone.eval()
        return self

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        h = self.drop(self.relu(self.fc7(f)))
        tool_logits = self.fc_tool(h)
        joint = torch.cat([h, tool_logits], dim=1)
        j = self.drop(self.relu(self.fc8(joint)))
        phase_logits = self.fc_phase(j)
        return phase_logits, tool_logits


def multitask_loss(phase_logits, tool_logits, phase_gt, tool_gt, a=1.0, b=1.0):
    ce = nn.CrossEntropyLoss()(phase_logits, phase_gt)
    bce = nn.BCEWithLogitsLoss()(tool_logits, tool_gt.float())
    return a * bce + b * ce, ce.detach(), bce.detach()
