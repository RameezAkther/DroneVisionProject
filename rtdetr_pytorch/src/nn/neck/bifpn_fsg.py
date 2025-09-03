import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core import register

def conv_bn_act(c_in, c_out, k=3, s=1, p=None):
    if p is None: p = k // 2
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, s, p, bias=False),
        nn.BatchNorm2d(c_out),
        nn.SiLU(inplace=True),
    )

class FSG(nn.Module):
    """Channel attention gate (Squeeze-Excitation style)"""
    def __init__(self, c, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, max(8, c // r), 1)
        self.fc2 = nn.Conv2d(max(8, c // r), c, 1)
        self.act = nn.SiLU()
        self.sig = nn.Sigmoid()
    def forward(self, x):
        w = self.avg(x)
        w = self.act(self.fc1(w))
        w = self.sig(self.fc2(w))
        return x * w

class BiFPNLayer(nn.Module):
    """A single BiFPN layer across P2..P5"""
    def __init__(self, c):
        super().__init__()
        self.p3_td = conv_bn_act(c, c, 3, 1)
        self.p2_td = conv_bn_act(c, c, 3, 1)
        self.p3_out = conv_bn_act(c, c, 3, 1)
        self.p4_out = conv_bn_act(c, c, 3, 1)
        self.p5_out = conv_bn_act(c, c, 3, 1)

        # learnable fusion weights (EfficientDet trick)
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))   # for P3_td = P3 + up(P4)
        self.w2 = nn.Parameter(torch.ones(2, dtype=torch.float32))   # for P2_td = P2 + up(P3_td)
        self.w3 = nn.Parameter(torch.ones(2, dtype=torch.float32))   # for P3_out = P3_td + down(P2_td)
        self.w4 = nn.Parameter(torch.ones(2, dtype=torch.float32))   # for P4_out = P4 + up/down blends
        self.w5 = nn.Parameter(torch.ones(2, dtype=torch.float32))   # for P5_out = P5 + down(P4_out)

    def _norm(self, w):  # positive, normalized
        w = torch.relu(w)
        return w / (w.sum(dim=0, keepdim=True) + 1e-6)

    def forward(self, P2, P3, P4, P5):
        # top-down
        w1 = self._norm(self.w1)
        P3_td = self.p3_td(w1[0] * P3 + w1[1] * F.interpolate(P4, scale_factor=2, mode='nearest'))
        w2 = self._norm(self.w2)
        P2_td = self.p2_td(w2[0] * P2 + w2[1] * F.interpolate(P3_td, scale_factor=2, mode='nearest'))

        # bottom-up
        w3 = self._norm(self.w3)
        P3_out = self.p3_out(w3[0] * P3_td + w3[1] * F.max_pool2d(P2_td, kernel_size=2))
        w4 = self._norm(self.w4)
        P4_out = self.p4_out(w4[0] * P4 + w4[1] * F.max_pool2d(P3_out, kernel_size=2))
        w5 = self._norm(self.w5)
        P5_out = self.p5_out(w5[0] * P5 + w5[1] * F.max_pool2d(P4_out, kernel_size=2))

        return P2_td, P3_out, P4_out, P5_out

@register
class BiFPN_FSG(nn.Module):
    """
    Neck that takes [P2,P3,P4,P5] (same channels) and returns the same,
    after N BiFPN layers + optional Feature Selection Gate per level.
    """
    def __init__(self, in_channels=256, num_layers=1, use_fsg=True):
        super().__init__()
        self.layers = nn.ModuleList([BiFPNLayer(in_channels) for _ in range(num_layers)])
        self.use_fsg = use_fsg
        if use_fsg:
            self.fsg = nn.ModuleList([FSG(in_channels) for _ in range(4)])

    def forward(self, feats):
        P2, P3, P4, P5 = feats
        for layer in self.layers:
            P2, P3, P4, P5 = layer(P2, P3, P4, P5)
        if self.use_fsg:
            P2, P3, P4, P5 = [g(f) for g, f in zip(self.fsg, [P2, P3, P4, P5])]
        return [P2, P3, P4, P5]
