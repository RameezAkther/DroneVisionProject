import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from collections import OrderedDict
from src.core import register

__all__ = ["ConvNeXtBackbone"]

def _freeze_norm(m: nn.Module):
    # Optional: keep BatchNorm frozen like PResNet does
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.LayerNorm)):
            for p in mod.parameters():
                p.requires_grad = False
    return m

@register
class ConvNeXtBackbone(nn.Module):
    """
    Wrapper around timm ConvNeXt that returns {P2, P3, P4, P5} feature maps.
    Strides: [4, 8, 16, 32]
    Channels (tiny/base): [96, 192, 384, 768] -> we project to 'out_channels' (256) with 1x1 convs.
    """
    def __init__(
        self,
        variant: str = "convnext_tiny.in12k_ft_in1k",  # or "convnext_tiny.fb_in22k_in1k"
        pretrained: bool = True,
        freeze_at: int = -1,         # 0: stem, 1: stage1, ... 4: stage4
        freeze_norm: bool = True,
        out_channels: int = 256,     # neck/encoder expect same dims
        return_idx = [0,1,2,3],      # P2..P5
    ):
        super().__init__()

        # Create ConvNeXt from timm
        self.backbone = timm.create_model(
            variant, pretrained=pretrained, features_only=True,
            out_indices=(0,1,2,3)  # C2..C5 features
        )
        feat_info = self.backbone.feature_info   # channels/stride info from timm
        self.strides = [fi["reduction"] for fi in feat_info]
        self.in_channels = [fi["num_chs"] for fi in feat_info]

        # 1x1 projections to a unified dim (like FPN)
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in self.in_channels
        ])

        self.return_idx = return_idx
        self.out_channels = [out_channels for _ in return_idx]
        self.out_strides = [self.strides[i] for i in return_idx]

        # Optional freezing
        if freeze_at >= 0:
            # freeze stem + first 'freeze_at' stages
            # features_only model exposes stages inside backbone.stages
            # simplest: freeze all
            for p in self.backbone.parameters():
                p.requires_grad = False
        if freeze_norm:
            _freeze_norm(self.backbone)

    def forward(self, x):
        feats = self.backbone(x)  # list: [C2, C3, C4, C5]
        outs = []
        for i, f in enumerate(feats):
            f = self.lateral[i](f)  # to out_channels
            if i in self.return_idx:
                outs.append(f)
        return outs  # [P2,P3,P4,P5] each: [B, out_channels, H/stride, W/stride]
