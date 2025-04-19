import torch
from torch import nn
import torch.nn.functional as F
from performer_pytorch import Performer

from utils import FeedForward, DECOUPLED


class STSAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        # STE
        self.ste = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=1),
            nn.Conv3d(1, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Sigmoid()
        )

        # DSA
        self.ca_conv = nn.Conv3d(1, channels // reduction, kernel_size=1)  # 修改为单通道输入
        self.ca_bn = nn.BatchNorm3d(channels // reduction)
        self.ca_act = nn.ReLU()

        self.h_conv = nn.Conv3d(channels // reduction, channels, kernel_size=1)
        self.w_conv = nn.Conv3d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, T, H, W = x.shape

        m1 = self.ste(x)

        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=4, keepdim=True)

        x_h_pool = x_h.mean(dim=1, keepdim=True)
        x_w_pool = x_w.mean(dim=1, keepdim=True)

        h_attn = self.ca_act(self.ca_bn(self.ca_conv(x_h_pool)))
        h_attn = self.sigmoid(self.h_conv(h_attn))

        w_attn = self.ca_act(self.ca_bn(self.ca_conv(x_w_pool)))
        w_attn = self.sigmoid(self.w_conv(w_attn))
        x=x * m1 * h_attn * w_attn

        return x

class BottleneckBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=0.5):
        super().__init__()
        mid_ch = int(out_ch * expansion)
        self.conv1 = nn.Conv3d(in_ch, mid_ch, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(mid_ch)
        self.conv2 = nn.Conv3d(mid_ch, mid_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(mid_ch)
        self.conv3 = nn.Conv3d(mid_ch, out_ch, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU()

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm3d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity
        out = self.relu(out)
        return out


class LGSModule(nn.Module):

    def __init__(self, channels, groups=2):
        super().__init__()
        self.groups = groups
        self.split_ch = channels // groups
        self.gate_conv = nn.Conv3d(channels, channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def shift(self, x, direction):
        if direction == 'forward':
            return F.pad(x[:, :, 1:], (0, 0, 0, 0, 0, 1), mode='constant')
        elif direction == 'backward':
            return F.pad(x[:, :, :-1], (0, 0, 0, 0, 1, 0), mode='constant')
        else:
            return x

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_group = x.view(B, self.groups, self.split_ch, T, H, W)

        x_shift_forward = self.shift(x_group[:, 0], 'forward')
        x_shift_backward = self.shift(x_group[:, 1], 'backward')

        gate = self.tanh(self.gate_conv(x))
        x_fused = gate * torch.cat([x_shift_forward, x_shift_backward], dim=1)
        return x_fused + x

class STGM(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bottle = BottleneckBlock(in_ch, out_ch)
        self.gsf = LGSModule(out_ch)
        self.sta = STSAttention(out_ch)

    def forward(self, x):
        x_re = self.sta(x)
        x = self.bottle(x)
        x = self.gsf(x)
        x = x+ x_re
        return x


# ---------------------- 整体模型 ----------------------
class Model(nn.Module):
    def __init__(
            self,
            *,
            dropout=0.4,
            attn_dropout=0.1,
            ff_mult=1,
    ):
        super().__init__()
        dims = (192,128 ,128)
        self.init_dim, self.conv_dim, last_dim = dims
        self.attn_dropout = attn_dropout
        self.frontend = nn.Sequential(
            nn.Conv3d(self.init_dim, self.conv_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(self.conv_dim),
            nn.GELU(),
            STGM(self.conv_dim, self.conv_dim),
            STGM(self.conv_dim, self.conv_dim),
            # STAB(self.conv_dim, self.conv_dim),

        )

        self.norm = nn.LayerNorm(last_dim)
        self.fc1 = nn.Linear(last_dim, 1)
        self.drop_out = nn.Dropout(dropout)
        self.pooling = nn.AdaptiveMaxPool3d((1, 1, 1))

    def forward(self, x):
        x = self.frontend(x)

        x = self.pooling(x).squeeze()
        x = self.drop_out(x)
        x = self.norm(x)
        logits = self.fc1(x)
        return logits,x


