"""R_PMNN model and standalone runner. / R_PMNN 模型及独立运行入口。

The network jointly predicts saturation, pore pressure, effective stress, and
slope stability. / 网络联合预测饱和度、孔隙压力、有效应力和边坡稳定性。

# 本代码用于固定库水位升降情况下，时间变量引导下的边坡饱和度、渗透压、以及稳定性全过程全面预测
# 输入：input1：初始边坡非饱和状态图，
#      input2：库水位升降控制信息，
#      input3：初始边坡孔隙压力图，
#      input4：初始边坡应力图，
# 输出：output1：升降时间T后的边坡饱和状态图像，
#      output2：升降时间T后的边坡孔隙压力图像
#      output3：升降时间T后的边坡应力图像
#      output5：升降时间T后的边坡稳定性
# Time：2026-06-23
# Email：fpf0103@163.com & 571428374@qq.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import argparse
import random
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from einops import rearrange
import numbers

# 显卡以及CUDA是否可用
if torch.cuda.is_available():
    print("GPU version installed.")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CPU version installed.")

# All image fields are stored as RGB PNGs and trained in normalized [0, 1] space.
transform = transforms.Compose([transforms.ToTensor()])

SLOPE_FEATURE_COLUMNS = (
    "INITIAL_WATER_LEVEL",
    "DIRECTION",
    "RATE",
    "DURATION",
    "SIGNED_RATE",
    "DELTA_WATER_LEVEL",
    "FINAL_WATER_LEVEL",
)
SLOPE_FEATURE_DIM = len(SLOPE_FEATURE_COLUMNS)
PORE_PRESSURE_MIN = -400000.0
PORE_PRESSURE_MAX = 200000.0
EFFECTIVE_STRESS_MIN = 0.0
EFFECTIVE_STRESS_MAX = 300000.0
FEATURE_NORMALIZATION_STATS = None


# 基本模块定义
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class GSAU(nn.Module):
    def __init__(self, n_feats, drop=0.0, k=2, squeeze_factor=15, attn='GLKA'):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        # Ghost Expand
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut

class MLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = 2 * n_feats
        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        # Multiscale Large Kernel Attention
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 4, n_feats // 4, 7, 1, 7 // 2, groups=n_feats // 4),
            nn.Conv2d(n_feats // 4, n_feats // 4, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // 4, dilation=4),
            nn.Conv2d(n_feats // 4, n_feats // 4, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 4, n_feats // 4, 5, 1, 5 // 2, groups=n_feats // 4),
            nn.Conv2d(n_feats // 4, n_feats // 4, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // 4, dilation=3),
            nn.Conv2d(n_feats // 4, n_feats // 4, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 4, n_feats // 4, 3, 1, 1, groups=n_feats // 4),
            nn.Conv2d(n_feats // 4, n_feats // 4, 5, stride=1, padding=(5 // 2) * 2, groups=n_feats // 4, dilation=2),
            nn.Conv2d(n_feats // 4, n_feats // 4, 1, 1, 0))
        self.LKA1 = nn.Sequential(
            nn.Conv2d(n_feats // 4, n_feats // 4, 1, 1, 0, groups=n_feats // 4),
            nn.Conv2d(n_feats // 4, n_feats // 4, 3, stride=1, padding=(3 // 2) * 2, groups=n_feats // 4, dilation=2),
            nn.Conv2d(n_feats // 4, n_feats // 4, 1, 1, 0))

        self.X1 = nn.Conv2d(n_feats // 4, n_feats // 4, 1, 1, 0, groups=n_feats // 4)
        self.X3 = nn.Conv2d(n_feats // 4, n_feats // 4, 3, 1, 1, groups=n_feats // 4)
        self.X5 = nn.Conv2d(n_feats // 4, n_feats // 4, 5, 1, 5 // 2, groups=n_feats // 4)
        self.X7 = nn.Conv2d(n_feats // 4, n_feats // 4, 7, 1, 7 // 2, groups=n_feats // 4)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3, a_4 = torch.chunk(a, 4, dim=1)
        a = torch.cat([self.LKA1(a_1) * self.X1(a_1), self.LKA3(a_2) * self.X3(a_2), self.LKA5(a_3) * self.X5(a_3),
                       self.LKA7(a_4) * self.X7(a_4)], dim=1)
        x = self.proj_last(x * a) * self.scale + shortcut
        return x
 # MAB
class MAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.LKA = MLKA(n_feats)
        self.LFE = GSAU(n_feats)
    def forward(self, x):
        # large kernel attention
        x = self.LKA(x)
        # local feature extraction
        x = self.LFE(x)
        return x

norm_dict = {'BATCH': nn.BatchNorm2d, 'INSTANCE': nn.InstanceNorm2d, 'GROUP': nn.GroupNorm, 'LAYER': nn.LayerNorm}
class Identity(nn.Module):
    """
    Identity mapping for building a residual connection
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
class ConvNorm(nn.Module):
    """
    Convolution and normalization
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, leaky=True, norm='BATCH', activation=True):
        super().__init__()
        # determine basic attributes
        self.norm_type = norm
        padding = (kernel_size - 1) // 2

        # activation, support PReLU and common ReLU
        if activation:
            self.act = nn.PReLU() if leaky else nn.ReLU(inplace=True)
        else:
            self.act = None

        # instantiate layers
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        norm_layer = norm_dict[norm]
        if norm in ['BATCH', 'INSTANCE', 'LAYER']:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = norm_layer(4, in_channels)

    def basic_forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

    def group_forward(self, x):
        x = self.norm(x)
        if self.act:
            x = self.act(x)
        x = self.conv(x)
        return x

    def forward(self, x):
        if self.norm_type in ['BATCH', 'INSTANCE', 'LAYER']:
            return self.basic_forward(x)
        else:
            return self.group_forward(x)

class ResBlock(nn.Module):
    """
    Residual blocks
    """
    def __init__(self, in_channels, out_channels, stride=1, use_dropout=False, leaky=False, norm='INSTANCE'):
        super().__init__()
        self.norm_type = norm
        self.act = nn.PReLU() if leaky else nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.1) if use_dropout else None

        self.conv1 = ConvNorm(in_channels, out_channels, 3, stride, leaky, norm, True)
        self.conv2 = ConvNorm(out_channels, out_channels, 3, 1, leaky, norm, False)

        need_map = in_channels != out_channels or stride != 1
        self.id = ConvNorm(in_channels, out_channels, 1, stride, leaky, norm, False) if need_map else Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        identity = self.id(identity)

        out = out + identity
        if self.norm_type != 'GROUP':
            out = self.act(out)

        if self.dropout:
            out = self.dropout(out)
        return out

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)

        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNormTransformer(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNormTransformer, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNormTransformer(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNormTransformer(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

# 时间序列特征模块定义
class FeatureModule(nn.Module):
    def __init__(self):
        super(FeatureModule, self).__init__()
        self.fc = nn.Linear(SLOPE_FEATURE_DIM, 20 * 35 * 8)
        self.relu = nn.ReLU()
        self.reshape = lambda x: x.view(-1, 8, 20, 35)

        self.ConvNorm1 = ConvNorm(in_channels=8, out_channels=8, kernel_size=3, stride=1, leaky=False, norm='INSTANCE', activation=True)
        self.encoder_level = nn.Sequential(*[
            TransformerBlock(dim=8, num_heads=1, ffn_expansion_factor=2.66, bias=False,
                             LayerNorm_type='WithBias') for i in range(4)])

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
            self.encoder_level,
            nn.InstanceNorm2d(8),
            nn.ReLU(),
        )
        self.ConvNorm2 = ConvNorm(in_channels=8, out_channels=16, kernel_size=3, stride=1, leaky=False, norm='INSTANCE',
                                  activation=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.reshape(x)
        x = self.ConvNorm1(x)
        x = self.conv_blocks(x)
        x = self.ConvNorm2(x)
        return x
class FeatureModuleStab(nn.Module):
    def __init__(self):
        super(FeatureModuleStab, self).__init__()
        self.fc = nn.Linear(SLOPE_FEATURE_DIM, 40 * 70 * 2)
        self.relu = nn.ReLU()
        self.reshape = lambda x: x.view(-1, 2, 40, 70)

        self.ConvNorm1 = ConvNorm(in_channels=2, out_channels=8, kernel_size=3, stride=1,
                                  leaky=False, norm='INSTANCE', activation=True)
        self.encoder_level = nn.Sequential(*[
            TransformerBlock(dim=8, num_heads=1, ffn_expansion_factor=2.66, bias=False,
                             LayerNorm_type='WithBias') for i in range(4)])

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
            self.encoder_level,
            nn.InstanceNorm2d(8),
            nn.ReLU(),
        )
        self.ConvNorm2 = ConvNorm(in_channels=8, out_channels=16, kernel_size=3, stride=1,
                                  leaky=False, norm='INSTANCE', activation=True)
        self.fc2 = nn.Sequential(
            nn.Linear(SLOPE_FEATURE_DIM, 40 * 70 * 2),
            nn.ReLU(),
            nn.Linear(40 * 70 * 2, 2),
        )

    def forward(self, value):
        x = self.fc(value)
        x = self.relu(x)
        x = self.reshape(x)
        x = self.ConvNorm1(x)
        x = self.conv_blocks(x)
        x1 = self.ConvNorm2(x)
        x2 = self.fc2(value)

        return x1,x2

# 孔隙压力初始条件化控制模块定义
class PorePressureControlModule(nn.Module):
    def __init__(self):
        super(PorePressureControlModule, self).__init__()
        # Encoder layers
        self.EncoderConvNorm1 = ConvNorm(in_channels=3, out_channels=16, kernel_size=3, stride=1, leaky=False,
                                         norm='INSTANCE',
                                         activation=True)
        self.EncoderMaxPool = nn.MaxPool2d(2, stride=2)
        self.EncoderConvNorm2 = ResBlock(in_channels=16, out_channels=16, stride=1, leaky=False,
                                         norm='INSTANCE')
        self.MAB = MAB(16)

        # Bridge layers
        self.bridge = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.EncoderConvNorm1(x)
        x1 = self.EncoderMaxPool(x)
        x = self.EncoderConvNorm2(x1)
        x = self.MAB(x)
        x2 = self.EncoderMaxPool(x)
        x = self.EncoderConvNorm2(x2)
        x = self.MAB(x)
        x3 = self.EncoderMaxPool(x)
        x = self.EncoderConvNorm2(x3)
        x = self.MAB(x)
        x4 = self.EncoderMaxPool(x)

        x = self.bridge(x4)
        return x, x2

# 应力初始条件化控制模块定义
class StressControlModule(nn.Module):
    def __init__(self):
        super(StressControlModule, self).__init__()
        # Encoder layers
        self.EncoderConvNorm1 = ConvNorm(in_channels=3, out_channels=16, kernel_size=3, stride=1, leaky=False,
                                         norm='INSTANCE',
                                         activation=True)
        self.EncoderMaxPool = nn.MaxPool2d(2, stride=2)
        self.EncoderConvNorm2 = ResBlock(in_channels=16, out_channels=16, stride=1, leaky=False,
                                         norm='INSTANCE')

        self.MAB = MAB(16)

        # Bridge layers
        self.bridge = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.EncoderConvNorm1(x)
        x1 = self.EncoderMaxPool(x)
        x = self.EncoderConvNorm2(x1)
        x = self.MAB(x)
        x2 = self.EncoderMaxPool(x)
        x = self.EncoderConvNorm2(x2)
        x = self.MAB(x)
        x3 = self.EncoderMaxPool(x)
        x = self.EncoderConvNorm2(x3)
        x = self.MAB(x)
        x4 = self.EncoderMaxPool(x)

        x = self.bridge(x4)
        return x, x2

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(gate_channels, gate_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(gate_channels // reduction_ratio, gate_channels, kernel_size=1),
            nn.Sigmoid()  # 输出0-1的权重
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),  # 用7x7卷积捕获空间关系
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_attention(x)
        x_channel_att = x * channel_att  # 通道加权

        # 空间注意力
        avg_out = torch.mean(x_channel_att, dim=1, keepdim=True)  # 通道平均
        max_out, _ = torch.max(x_channel_att, dim=1, keepdim=True)  # 通道最大
        spatial_att_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_att_input)
        x_att = x_channel_att * spatial_att  # 空间加权

        return x_att

# 主网络结构
class R_PMNN(nn.Module):
    """Joint physical-field and stability predictor. / 物理场与稳定性联合预测网络。"""

    def __init__(self):
        super().__init__()

        # Encoder layers
        self.EncoderConvNorm1 = ConvNorm(in_channels=3, out_channels=16, kernel_size=3, stride=1, leaky=False, norm='INSTANCE',
                                  activation=True)
        self.EncoderMaxPool = nn.MaxPool2d(2, stride=2)
        self.EncoderConvNorm2 = ResBlock(in_channels=16, out_channels=16, stride=1, leaky=False,
                                         norm='INSTANCE')
        self.MAB = MAB(16)

        # Bridge layers
        self.bridge = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Decoder layers
        self.decoderConvTrans = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
        )
        self.decodeConvNorm1 = ConvNorm(in_channels=16, out_channels=16, kernel_size=3, stride=1, leaky=False,
                                         norm='INSTANCE',
                                         activation=True)

        # Task-specific output heads keep the three physical fields from sharing
        # a single RGB mapping.
        self.final_conv_sat = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.final_conv_pore = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.final_conv_stress = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

        self.FeatureModule = FeatureModule()
        self.FeatureModuleStab = FeatureModuleStab()
        self.PorePressureControlModule  = PorePressureControlModule()
        self.StressControlModule = StressControlModule()

        # 稳定性预测部分架构
        self.stabilyBasicModule = nn.Sequential(
            ConvNorm(in_channels=16, out_channels=32, kernel_size=3, stride=1, leaky=False,
                     norm='INSTANCE',
                     activation=True),
            nn.MaxPool2d(2, stride=2),
            CBAM(32),
            ConvNorm(in_channels=32, out_channels=16, kernel_size=3, stride=1, leaky=False,
                     norm='INSTANCE',
                     activation=True),
            nn.MaxPool2d(2, stride=2),
            CBAM(16),
        )

        # 稳定性部分全连接以及池化层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 40 * 70, 128)
        self.fc2 = nn.Linear(128, 2)
        self.fc3 = nn.Linear(2, 1)

        # Shared uncertainty terms / 图像场与稳定性任务的共享不确定性参数
        self.log_var1 = nn.Parameter(torch.zeros(1))
        self.log_var2 = nn.Parameter(torch.zeros(1))

    def compute_Saturability_PorePressure(self,Sr):
        m = 1
        n = 2
        alpha = 1
        waterp = 1000
        g = 9.81
        # 避免除零错误
        S_clamp = torch.clamp(Sr, 0.05, 0.99)

        term = torch.pow(S_clamp, -1 / m) - 1  # 计算 Sr^(-1/m) - 1
        term = torch.pow(term, 1 / n)  # 计算 (term)^(1/n)
        psi_prime = (-1 / alpha) * term  # 乘以 -1/alpha

        return waterp * g * psi_prime

    def forward(self, x, featureinit, PorePressure, Stress):
        x = self.EncoderConvNorm1(x)
        x1 = self.EncoderMaxPool(x)
        x = self.EncoderConvNorm2(x1)
        x = self.MAB(x)
        x2 = self.EncoderMaxPool(x)
        x = self.EncoderConvNorm2(x2)
        x = self.MAB(x)
        x3 = self.EncoderMaxPool(x)
        x = self.EncoderConvNorm2(x3)
        x = self.MAB(x)
        x4 = self.EncoderMaxPool(x)

        x = self.bridge(x4)

        feature = self.FeatureModule(featureinit.float())
        PorePressureControl, pore_skip2 = self.PorePressureControlModule(PorePressure)
        StressControl, stress_skip2 = self.StressControlModule(Stress)
        # 融合特征
        temp = x * feature
        x = x + temp
        PorePressureControl = (PorePressureControl + PorePressureControl * feature + self.compute_Saturability_PorePressure(
                                      x)) / 2
        StressControl = StressControl + (StressControl * feature + x * PorePressureControl) / 2

        x = self.decoderConvTrans(x)
        x = self.decodeConvNorm1(x)
        feature = self.decoderConvTrans(feature)
        PorePressureControl = self.decoderConvTrans(PorePressureControl)
        PorePressureControl = self.decodeConvNorm1(PorePressureControl)
        StressControl = self.decoderConvTrans(StressControl)
        StressControl = self.decodeConvNorm1(StressControl)


        x = self.decoderConvTrans(x)
        x = self.decodeConvNorm1(x)
        PorePressureControl = self.decoderConvTrans(PorePressureControl)
        PorePressureControl = self.decodeConvNorm1(PorePressureControl)
        StressControl = self.decoderConvTrans(StressControl)
        StressControl = self.decodeConvNorm1(StressControl)

        x = x + x2
        if bool(globals().get("FIELD_BRANCH_SKIP_ENABLED", True)):
            PorePressureControl = PorePressureControl + pore_skip2
            StressControl = StressControl + stress_skip2

        x = self.decoderConvTrans(x)
        x = self.decodeConvNorm1(x)
        PorePressureControl = self.decoderConvTrans(PorePressureControl)
        PorePressureControl = self.decodeConvNorm1(PorePressureControl)
        StressControl = self.decoderConvTrans(StressControl)
        StressControl = self.decodeConvNorm1(StressControl)

        feature = self.decoderConvTrans(feature)
        feature = self.decoderConvTrans(feature)

        temp = x * feature
        x = x + temp
        PorePressureControl = (PorePressureControl + PorePressureControl * feature + self.compute_Saturability_PorePressure(
                                      x)) / 2
        StressControl = StressControl + (StressControl * feature + x * PorePressureControl) / 2

        # 稳定性预测部分支路连接点
        Stabilization = (PorePressureControl + StressControl).detach()
        Stabilization = (Stabilization - Stabilization.mean()) / (Stabilization.std() + 1e-6)

        x = self.decoderConvTrans(x)
        x = self.decodeConvNorm1(x)
        PorePressureControl = self.decoderConvTrans(PorePressureControl)
        PorePressureControl = self.decodeConvNorm1(PorePressureControl)
        StressControl = self.decoderConvTrans(StressControl)
        StressControl = self.decodeConvNorm1(StressControl)

        x = self.final_conv_sat(x)
        x = self.sigmoid(x)
        pore_logits = self.final_conv_pore(PorePressureControl)
        stress_logits = self.final_conv_stress(StressControl)
        pore_delta = torch.tanh(pore_logits)
        stress_delta = torch.tanh(stress_logits)
        if bool(globals().get("FIELD_RESIDUAL_ENABLED", True)):
            pore_scale = float(globals().get("PORE_RESIDUAL_SCALE", 0.25))
            stress_scale = float(globals().get("STRESS_RESIDUAL_SCALE", 0.25))
            PorePressureControl = torch.clamp(PorePressure + pore_scale * pore_delta, 0.0, 1.0)
            StressControl = torch.clamp(Stress + stress_scale * stress_delta, 0.0, 1.0)
        else:
            PorePressureControl = self.sigmoid(pore_logits)
            StressControl = self.sigmoid(stress_logits)

        # 稳定性预测部分架构
        fs, fv = self.FeatureModuleStab(featureinit.float())

        Stabilization = self.stabilyBasicModule(Stabilization)

        Stabilization = Stabilization + fs

        Stabilization = self.flatten(Stabilization)

        Stabilization = F.relu(self.fc1(Stabilization))

        Stabilization = self.fc2(Stabilization) + self.flatten(fv)
        Stabilization = F.relu(Stabilization)
        Stabilization = self.fc3(Stabilization)

        return x, PorePressureControl, StressControl, Stabilization

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 损失函数
criterion = nn.MSELoss()

def _loss_config_value(name, default):
    return float(globals().get(name, default))


def image_task_weight(task_name):
    weights = globals().get("IMAGE_LOSS_WEIGHTS", {})
    return float(weights.get(task_name, 1.0))


def gradient_consistency_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return criterion(pred_dx, target_dx) + criterion(pred_dy, target_dy)


def total_variation_loss(field):
    tv_x = torch.mean(torch.abs(field[:, :, :, 1:] - field[:, :, :, :-1]))
    tv_y = torch.mean(torch.abs(field[:, :, 1:, :] - field[:, :, :-1, :]))
    return tv_x + tv_y


def image_field_loss(pred, target, task_name):
    loss = criterion(pred, target)
    loss = loss + _loss_config_value("GRADIENT_LOSS_WEIGHT", 0.2) * gradient_consistency_loss(pred, target)
    if task_name in ("pore_pressure", "effective_stress"):
        loss = loss + _loss_config_value("TV_LOSS_WEIGHT", 0.05) * total_variation_loss(pred)
    return image_task_weight(task_name) * loss


def rgb_scalar(field):
    return 0.299 * field[:, 0:1] + 0.587 * field[:, 1:2] + 0.114 * field[:, 2:3]


_SOIL_MASK_CACHE = {}


def rgb_to_physical(field, vmin, vmax):
    gray = rgb_scalar(field)
    return vmin + gray.clamp(0.0, 1.0) * (vmax - vmin)


def feature_matrix(feature_values):
    return feature_values.float().view(feature_values.size(0), -1)


def physical_feature_matrix(feature_values):
    features = feature_matrix(feature_values)
    stats = globals().get("FEATURE_NORMALIZATION_STATS")
    if not stats:
        return features
    columns = stats.get("feature_columns", [])
    if features.size(1) != len(columns):
        return features
    mean = torch.tensor(
        [stats["mean"][column] for column in columns],
        dtype=features.dtype,
        device=features.device,
    )
    std = torch.tensor(
        [stats["std"][column] for column in columns],
        dtype=features.dtype,
        device=features.device,
    )
    return features * std + mean


def duration_from_features(feature_values):
    features = physical_feature_matrix(feature_values)
    if features.size(1) >= 4:
        return features[:, 3].clamp_min(1.0)
    return torch.ones(features.size(0), dtype=features.dtype, device=features.device)


def load_soil_mask_like(field):
    mask_path = globals().get("SOIL_MASK_PATH")
    if not mask_path:
        return torch.ones(
            (1, 1, field.size(-2), field.size(-1)),
            dtype=field.dtype,
            device=field.device,
        )
    key = (str(mask_path), field.size(-2), field.size(-1), str(field.device), str(field.dtype))
    if key in _SOIL_MASK_CACHE:
        return _SOIL_MASK_CACHE[key]
    if not os.path.exists(mask_path):
        mask = torch.ones(
            (1, 1, field.size(-2), field.size(-1)),
            dtype=field.dtype,
            device=field.device,
        )
    else:
        arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        mask = torch.from_numpy((arr > 0.5).astype(np.float32)).view(1, 1, arr.shape[0], arr.shape[1])
        mask = mask.to(device=field.device, dtype=field.dtype)
        if mask.shape[-2:] != field.shape[-2:]:
            mask = F.interpolate(mask, size=field.shape[-2:], mode="nearest")
    _SOIL_MASK_CACHE[key] = mask
    return mask


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def compute_laplacian(field, dx=1.0, dy=1.0):
    lap_x = (field[:, :, :, 2:] - 2.0 * field[:, :, :, 1:-1] + field[:, :, :, :-2]) / (dx * dx)
    lap_y = (field[:, :, 2:, :] - 2.0 * field[:, :, 1:-1, :] + field[:, :, :-2, :]) / (dy * dy)
    lap_x = F.pad(lap_x, (1, 1, 0, 0), mode="replicate")
    lap_y = F.pad(lap_y, (0, 0, 1, 1), mode="replicate")
    return lap_x + lap_y


def physics_ramp_weight(weight_name, default, epoch):
    weight = _loss_config_value(weight_name, default)
    if weight <= 0.0:
        return 0.0
    warmup = int(globals().get("PHYSICS_WARMUP_EPOCHS", 20))
    if epoch is None or warmup <= 0:
        return weight
    if epoch < warmup:
        return 0.0
    ramp = min(1.0, (epoch - warmup + 1) / max(warmup, 1))
    return weight * ramp


def delta_physics_loss(pred_pore, pred_stress, target_pore, target_stress, initial_pore, initial_stress):
    mask = load_soil_mask_like(pred_pore)
    pore_scale = PORE_PRESSURE_MAX - PORE_PRESSURE_MIN
    stress_scale = EFFECTIVE_STRESS_MAX - EFFECTIVE_STRESS_MIN
    pred_delta_pore = rgb_to_physical(pred_pore, PORE_PRESSURE_MIN, PORE_PRESSURE_MAX) - rgb_to_physical(
        initial_pore, PORE_PRESSURE_MIN, PORE_PRESSURE_MAX
    )
    target_delta_pore = rgb_to_physical(target_pore, PORE_PRESSURE_MIN, PORE_PRESSURE_MAX) - rgb_to_physical(
        initial_pore, PORE_PRESSURE_MIN, PORE_PRESSURE_MAX
    )
    pred_delta_stress = rgb_to_physical(pred_stress, EFFECTIVE_STRESS_MIN, EFFECTIVE_STRESS_MAX) - rgb_to_physical(
        initial_stress, EFFECTIVE_STRESS_MIN, EFFECTIVE_STRESS_MAX
    )
    target_delta_stress = rgb_to_physical(target_stress, EFFECTIVE_STRESS_MIN, EFFECTIVE_STRESS_MAX) - rgb_to_physical(
        initial_stress, EFFECTIVE_STRESS_MIN, EFFECTIVE_STRESS_MAX
    )
    pore_loss = masked_mean(((pred_delta_pore - target_delta_pore) / pore_scale).pow(2), mask)
    stress_loss = masked_mean(((pred_delta_stress - target_delta_stress) / stress_scale).pow(2), mask)
    return pore_loss + stress_loss


def pore_diffusion_residual_loss(pred_pore, initial_pore, feature_values):
    mask = load_soil_mask_like(pred_pore)
    pressure_scale = PORE_PRESSURE_MAX - PORE_PRESSURE_MIN
    pred_pressure = rgb_to_physical(pred_pore, PORE_PRESSURE_MIN, PORE_PRESSURE_MAX)
    initial_pressure = rgb_to_physical(initial_pore, PORE_PRESSURE_MIN, PORE_PRESSURE_MAX)
    duration = duration_from_features(feature_values).view(-1, 1, 1, 1)
    normalized_pressure = pred_pressure / pressure_scale
    pressure_rate = (pred_pressure - initial_pressure) / pressure_scale / duration
    diffusivity = float(globals().get("PORE_DIFFUSIVITY", 0.01))
    residual = pressure_rate - diffusivity * compute_laplacian(normalized_pressure)
    return masked_mean(residual.pow(2), mask)


# Multi-task weak-physics loss / 多任务弱物理损失
def modelLoss(net, XTrain1, XTrain2, XTrain3, XTrain4, Y1, Y2, Y3, Y5, epoch):
    XTrain1 = XTrain1.to(device)
    XTrain2 = XTrain2.to(device)
    XTrain3 = XTrain3.to(device)
    XTrain4 = XTrain4.to(device)
    Y1 = Y1.to(device)
    Y2 = Y2.to(device)
    Y3 = Y3.to(device)
    Y5 = Y5.to(device)

    # Forward pass: initial saturation, reservoir features, initial pore pressure, and initial stress.
    outputs1, outputs2, outputs3, outputs5 = net(XTrain1, XTrain2, XTrain3, XTrain4)

    # Data-driven image losses are augmented by weak physics penalties below.
    loss1 = image_field_loss(outputs1, Y1, "saturation")
    loss2 = image_field_loss(outputs2, Y2, "pore_pressure")
    loss3 = image_field_loss(outputs3, Y3, "effective_stress")
    loss5 = criterion(outputs5.view(-1).float(), Y5.float())
    loss_delta_physics = delta_physics_loss(outputs2, outputs3, Y2, Y3, XTrain3, XTrain4)
    loss_pore_diffusion = pore_diffusion_residual_loss(outputs2, XTrain3, XTrain2)
    physics_loss = (
        physics_ramp_weight("DELTA_PHYSICS_LOSS_WEIGHT", 0.0, epoch) * loss_delta_physics
        + physics_ramp_weight("PORE_DIFFUSION_LOSS_WEIGHT", 0.0, epoch) * loss_pore_diffusion
    )

    # Shared uncertainty weights balance the three image fields against stability.
    loss = (
        torch.exp(-net.log_var1) * (loss1 + loss2 + loss3)
        + net.log_var1
        + physics_loss
        + torch.exp(-net.log_var2) * float(globals().get("STABILITY_LOSS_WEIGHT", 1.0)) * loss5
        + net.log_var2
    )

    return loss, loss1, loss2, loss3, loss5
def train_model(datasetTrain, model_save_dir='models', batch_size=4, epochs=500, lr=0.01, checkpoint_prefix=None):
    if checkpoint_prefix is None:
        checkpoint_prefix = str(globals().get("CHECKPOINT_PREFIX", "R_PMNN"))

    dataloader = DataLoader(datasetTrain, batch_size=batch_size, shuffle=True)
    # 初始化模型
    model = R_PMNN()
    model.to(device)

    # 优化器
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
    # 训练
    num_epochs = epochs
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (x1, x2, x3, x4, y1, y2, y3, y5) in enumerate(dataloader):
            x1, x2, x3, x4, y1, y2, y3, y5 = (x1.to(device), x2.unsqueeze(1).to(device),
                                                  x3.to(device), x4.to(device),
                                                  y1.to(device), y2.to(device), y3.to(device),
                                                  y5.to(device))

            loss, loss1, loss2, loss3, loss5 = modelLoss(model, x1, x2, x3, x4, y1, y2, y3, y5, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{num_epochs}] Batch [{i + 1}],"
                  f" Loss: {loss.item():.4f}, Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f},"
                  f" Loss3: {loss3.item():.4f}, Loss5: {loss5.item():.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
        if (epoch + 1) % 100 == 0 or epoch + 1 == num_epochs:
            # 模型训练好之后的保存路径
            snapshot_path = model_save_dir
            os.makedirs(snapshot_path, exist_ok=True)
            save_mode_path = os.path.join(snapshot_path, f'{checkpoint_prefix}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), save_mode_path)

def value_to_Fr(outputs):
    """
    将 outputs 中的值四舍五入为整数索引，并从 Fr 数组中提取对应值。

    参数:
        outputs (np.ndarray 或 torch.Tensor): 输入的浮点数数组/张量

    返回:
        outputs1 (np.ndarray): 从 Fr 中提取的对应值
    """
    # 定义 Fr 数组（固定范围 0.98:0.00005:1.12）
    Fr = np.arange(0.93, 1.30, 0.0001)  # 2801 个点 1.1200001 确保包含 1.12

    # 处理输入类型（兼容 NumPy 和 PyTorch）
    if hasattr(outputs, 'numpy'):  # 如果是 PyTorch 张量
        outputs_np = outputs.detach().cpu().numpy()  # 转为 NumPy 数组
    else:
        outputs_np = np.array(outputs)  # 确保是 NumPy 数组

    # 1. 四舍五入取整，并限制索引范围
    indices = np.round(outputs_np).astype(int)
    indices = np.clip(indices, 0, len(Fr) - 1)  # 避免越界

    # 2. 从 Fr 中提取值
    outputs1 = Fr[indices]

    return outputs1

# =============================================================================
# Standalone Full R_PMNN runner
# =============================================================================

CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parent

FULL_FEATURE_COLUMNS = [
    "INITIAL_WATER_LEVEL",
    "DIRECTION",
    "RATE",
    "DURATION",
    "SIGNED_RATE",
    "DELTA_WATER_LEVEL",
    "FINAL_WATER_LEVEL",
]
FULL_LABEL_COLUMN = "StabilizationFactorIndex"
FULL_NAME_COLUMN = "name"
FULL_CHECKPOINT_PREFIX = "R_PMNN_full"
FULL_CHECKPOINT_DIR = CODE_DIR / "checkpoints" / "full"
FULL_OUTPUT_DIR = CODE_DIR / "results" / "full_standalone"
FULL_STATS_PATH = FULL_CHECKPOINT_DIR / "reservoir_feature_stats.json"

FULL_TRAIN_METADATA = WORKSPACE_DIR / "dataset" / "H15-JS5-T2-50.xlsx"
FULL_TRAIN_IMAGE_ROOT = WORKSPACE_DIR / "dataset"
FULL_PREDICT_METADATA = WORKSPACE_DIR / "dataset" / "testfull" / "H15-JS5-T4-40.xlsx"
FULL_PREDICT_IMAGE_ROOT = WORKSPACE_DIR / "dataset" / "testfull"

FULL_TRAIN_IMAGE_DIRS = {
    "x1": "SaturabilityTraincopy",
    "x3": "PorePressureTraincopy",
    "x4": "EffectiveStressTraincopy",
    "y1": "Saturabilitycopy",
    "y2": "PorePressurecopy",
    "y3": "EffectiveStresscopy",
}
FULL_PREDICT_IMAGE_DIRS = {
    "x1": "SaturabilityTraincopy",
    "x3": "PorePressureTraincopy",
    "x4": "EffectiveStressTraincopy",
    "y1": "Saturabilitycopy",
    "y2": "PorePressurecopy",
    "y3": "EffectiveStresscopy",
}
FULL_NAME_REPLACEMENTS = {
    "x3": [("S_", "P_")],
    "x4": [("S_", "E_")],
    "y2": [("S_", "P_")],
    "y3": [("S_", "E_")],
}
TASK_OUTPUT_DIRS = {
    "task1": "saturation",
    "task2": "pore_pressure",
    "task3": "effective_stress",
}


def set_random_seed(seed=60):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass


def configure_full_r_pmnn():
    """Apply the fixed full-model settings. / 应用完整模型的固定实验参数。"""

    global STABILITY_LOSS_WEIGHT
    global IMAGE_LOSS_WEIGHTS
    global GRADIENT_LOSS_WEIGHT
    global TV_LOSS_WEIGHT
    global SOIL_MASK_PATH
    global PHYSICS_WARMUP_EPOCHS
    global DELTA_PHYSICS_LOSS_WEIGHT
    global PORE_DIFFUSION_LOSS_WEIGHT
    global PORE_DIFFUSIVITY
    global FIELD_BRANCH_SKIP_ENABLED
    global FIELD_RESIDUAL_ENABLED
    global PORE_RESIDUAL_SCALE
    global STRESS_RESIDUAL_SCALE
    global CHECKPOINT_PREFIX

    STABILITY_LOSS_WEIGHT = 1.0
    IMAGE_LOSS_WEIGHTS = {"saturation": 1.0, "pore_pressure": 1.5, "effective_stress": 2.8}
    GRADIENT_LOSS_WEIGHT = 0.2
    TV_LOSS_WEIGHT = 0.05
    SOIL_MASK_PATH = str(WORKSPACE_DIR / "dataset" / "mask.png")
    PHYSICS_WARMUP_EPOCHS = 20
    DELTA_PHYSICS_LOSS_WEIGHT = 0.05
    PORE_DIFFUSION_LOSS_WEIGHT = 0.0001
    PORE_DIFFUSIVITY = 0.01
    FIELD_BRANCH_SKIP_ENABLED = True
    FIELD_RESIDUAL_ENABLED = True
    PORE_RESIDUAL_SCALE = 0.25
    STRESS_RESIDUAL_SCALE = 0.25
    CHECKPOINT_PREFIX = FULL_CHECKPOINT_PREFIX


def clean_image_name(value):
    return str(value).strip().strip("'").strip('"')


def transform_image_name(name, key):
    name = clean_image_name(name)
    for old, new in FULL_NAME_REPLACEMENTS.get(key, []):
        name = name.replace(old, new)
    return name


def add_reservoir_feature_columns(frame):
    frame = frame.copy()
    if "DURATION" not in frame.columns and "TIME" in frame.columns:
        frame["DURATION"] = frame["TIME"]
    frame["SIGNED_RATE"] = frame["DIRECTION"].astype(float) * frame["RATE"].astype(float)
    frame["DELTA_WATER_LEVEL"] = frame["SIGNED_RATE"].astype(float) * frame["DURATION"].astype(float)
    frame["FINAL_WATER_LEVEL"] = frame["INITIAL_WATER_LEVEL"].astype(float) + frame["DELTA_WATER_LEVEL"].astype(float)
    return frame


def compute_feature_stats(frame):
    values = frame[FULL_FEATURE_COLUMNS].astype(float)
    mean = values.mean()
    std = values.std(ddof=0).replace(0, 1.0)
    return {
        "feature_columns": list(FULL_FEATURE_COLUMNS),
        "mean": {column: float(mean[column]) for column in FULL_FEATURE_COLUMNS},
        "std": {column: float(std[column]) for column in FULL_FEATURE_COLUMNS},
    }


def save_feature_stats(stats):
    import json

    FULL_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FULL_STATS_PATH.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_feature_stats():
    import json

    with FULL_STATS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_feature_normalization(frame, stats):
    normalized = frame.copy()
    for column in FULL_FEATURE_COLUMNS:
        mean = float(stats["mean"][column])
        std = float(stats["std"][column]) or 1.0
        normalized[column] = (normalized[column].astype(float) - mean) / std
    return normalized


class FullAlignedSlopeDataset(Dataset):
    def __init__(self, frame, feature_frame, label_values, image_root, image_dirs):
        self.frame = frame.reset_index(drop=True)
        self.feature_values = feature_frame.reset_index(drop=True)[FULL_FEATURE_COLUMNS].astype(np.float32).to_numpy()
        self.label_values = np.asarray(label_values, dtype=np.float32)
        self.image_root = Path(image_root)
        self.image_dirs = image_dirs
        self.paths = {
            key: self._resolve_image_paths(key, self.image_root / rel_dir)
            for key, rel_dir in image_dirs.items()
        }

    def _resolve_image_paths(self, key, directory):
        paths = [directory / transform_image_name(name, key) for name in self.frame[FULL_NAME_COLUMN]]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            sample = "\n".join(missing[:5])
            raise FileNotFoundError(f"Missing image files in {directory}:\n{sample}")
        return paths

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        with Image.open(self.paths["x1"][idx]) as img:
            x1 = transform(img.convert("RGB"))
        with Image.open(self.paths["x3"][idx]) as img:
            x3 = transform(img.convert("RGB"))
        with Image.open(self.paths["x4"][idx]) as img:
            x4 = transform(img.convert("RGB"))
        with Image.open(self.paths["y1"][idx]) as img:
            y1 = transform(img.convert("RGB"))
        with Image.open(self.paths["y2"][idx]) as img:
            y2 = transform(img.convert("RGB"))
        with Image.open(self.paths["y3"][idx]) as img:
            y3 = transform(img.convert("RGB"))
        return x1, self.feature_values[idx], x3, x4, y1, y2, y3, self.label_values[idx]


def build_full_dataset(metadata_file, image_root, image_dirs, stats=None, fit_stats=False):
    frame = pd.read_excel(metadata_file, sheet_name="Sheet2")
    frame = add_reservoir_feature_columns(frame)
    missing = [
        column for column in FULL_FEATURE_COLUMNS + [FULL_LABEL_COLUMN, FULL_NAME_COLUMN]
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")
    if fit_stats:
        stats = compute_feature_stats(frame)
        save_feature_stats(stats)
    elif stats is None:
        stats = load_feature_stats()
    globals()["FEATURE_NORMALIZATION_STATS"] = stats
    feature_frame = apply_feature_normalization(frame, stats)
    label_values = frame[FULL_LABEL_COLUMN].astype(float).to_numpy()
    dataset = FullAlignedSlopeDataset(frame, feature_frame, label_values, image_root, image_dirs)
    return frame, dataset, stats


def image_to_uint8(tensor):
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def scalar_field_numpy(tensor):
    field = tensor.detach().cpu().float().numpy()
    if field.ndim == 3 and field.shape[0] >= 3:
        return 0.299 * field[0] + 0.587 * field[1] + 0.114 * field[2]
    if field.ndim == 3:
        return field.mean(axis=0)
    return field


def save_signed_field_map(tensor, path, limit):
    values = scalar_field_numpy(tensor)
    limit = max(float(limit), 1e-6)
    plt.imsave(str(path), values, cmap="coolwarm", vmin=-limit, vmax=limit)


def save_abs_field_map(tensor, path, limit):
    values = np.abs(scalar_field_numpy(tensor))
    limit = max(float(limit), 1e-6)
    plt.imsave(str(path), values, cmap="inferno", vmin=0.0, vmax=limit)


def delta_field_stats(prefix, pred_delta, true_delta):
    diff = pred_delta - true_delta
    diff_rgb = diff.detach().cpu().float().numpy()
    pred_scalar = scalar_field_numpy(pred_delta)
    true_scalar = scalar_field_numpy(true_delta)
    diff_scalar = pred_scalar - true_scalar
    return {
        f"{prefix}_pred_delta_mean": float(pred_scalar.mean()),
        f"{prefix}_true_delta_mean": float(true_scalar.mean()),
        f"{prefix}_delta_mean_error": float(diff_scalar.mean()),
        f"{prefix}_delta_mae": float(np.mean(np.abs(diff_scalar))),
        f"{prefix}_delta_mse": float(np.mean(diff_scalar ** 2)),
        f"{prefix}_delta_max_abs_error": float(np.max(np.abs(diff_scalar))),
        f"{prefix}_rgb_delta_mae": float(np.mean(np.abs(diff_rgb))),
    }


def numeric_values(predictions, truths):
    pred = np.round(value_to_Fr(predictions), 5)
    true = np.round(value_to_Fr(truths), 5)
    return np.asarray(pred, dtype=float).reshape(-1), np.asarray(true, dtype=float).reshape(-1)


def standalone_predict(dataset, frame, output_dir, model_path, batch_size=1):
    """Run the full model on one dataset split. / 在一个数据划分上运行完整模型。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = R_PMNN().to(device)
    try:
        state_dict = torch.load(str(model_path), weights_only=True, map_location=device)
    except TypeError:  # Compatibility with older PyTorch / 兼容旧版 PyTorch
        state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    prefix = "R_PMNN_full"
    metrics = {
        "task1_psnr": [], "task1_ssim": [],
        "task2_psnr": [], "task2_ssim": [],
        "task3_psnr": [], "task3_ssim": [],
        "task5_mse": [], "task5_mae": [],
    }
    prediction_rows = []
    sample_metric_rows = []
    explainability_rows = []
    sample_index = 0

    with torch.no_grad():
        for batch in dataloader:
            x1, x2, x3, x4, y1, y2, y3, y5 = batch
            x1 = x1.to(device)
            x2 = x2.unsqueeze(1).to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y3 = y3.to(device)
            y5 = y5.to(device)

            outputs1, outputs2, outputs3, outputs5 = model(x1, x2, x3, x4)
            pred_values, true_values = numeric_values(outputs5, y5)

            for offset in range(x1.size(0)):
                row = frame.iloc[sample_index + offset]
                stem = Path(clean_image_name(row[FULL_NAME_COLUMN])).stem

                for task, output in (("task1", outputs1[offset]), ("task2", outputs2[offset]), ("task3", outputs3[offset])):
                    task_dir = output_dir / f"{prefix}_{TASK_OUTPUT_DIRS[task]}"
                    task_dir.mkdir(parents=True, exist_ok=True)
                    plt.imsave(str(task_dir / f"prediction_{stem}.png"), image_to_uint8(output))

                sample_metrics = {}
                for task, output, target in (
                    ("task1", outputs1[offset], y1[offset]),
                    ("task2", outputs2[offset], y2[offset]),
                    ("task3", outputs3[offset], y3[offset]),
                ):
                    psnr_value = psnr_metric(output.unsqueeze(0), target.unsqueeze(0)).detach().cpu().item()
                    ssim_value = ssim_metric(output.unsqueeze(0), target.unsqueeze(0)).detach().cpu().item()
                    metrics[f"{task}_psnr"].append(psnr_value)
                    metrics[f"{task}_ssim"].append(ssim_value)
                    sample_metrics[f"{task}_psnr"] = psnr_value
                    sample_metrics[f"{task}_ssim"] = ssim_value

                task5_mse = float((pred_values[offset] - true_values[offset]) ** 2)
                task5_mae = float(abs(pred_values[offset] - true_values[offset]))
                metrics["task5_mse"].append(task5_mse)
                metrics["task5_mae"].append(task5_mae)
                sample_metrics["task5_mse"] = task5_mse
                sample_metrics["task5_mae"] = task5_mae
                sample_metrics["task5_error"] = float(pred_values[offset] - true_values[offset])

                pore_pred_delta = outputs2[offset] - x3[offset]
                pore_true_delta = y2[offset] - x3[offset]
                stress_pred_delta = outputs3[offset] - x4[offset]
                stress_true_delta = y3[offset] - x4[offset]
                explainability_stats = {}
                explainability_stats.update(delta_field_stats("pore", pore_pred_delta, pore_true_delta))
                explainability_stats.update(delta_field_stats("stress", stress_pred_delta, stress_true_delta))

                explainability_dir = output_dir / f"{prefix}_explainability"
                explainability_paths = (
                    ("pore_delta_pred", pore_pred_delta, 0.25, "signed"),
                    ("pore_delta_true", pore_true_delta, 0.25, "signed"),
                    ("pore_delta_abs_error", pore_pred_delta - pore_true_delta, 0.25, "abs"),
                    ("stress_delta_pred", stress_pred_delta, 0.25, "signed"),
                    ("stress_delta_true", stress_true_delta, 0.25, "signed"),
                    ("stress_delta_abs_error", stress_pred_delta - stress_true_delta, 0.25, "abs"),
                )
                for subdir, tensor, limit, mode in explainability_paths:
                    target_dir = explainability_dir / subdir
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path = target_dir / f"{subdir}_{stem}.png"
                    if mode == "signed":
                        save_signed_field_map(tensor, target_path, limit)
                    else:
                        save_abs_field_map(tensor, target_path, limit)

                result = {column: row[column] for column in FULL_FEATURE_COLUMNS}
                result[FULL_LABEL_COLUMN] = row[FULL_LABEL_COLUMN]
                result[FULL_NAME_COLUMN] = row[FULL_NAME_COLUMN]
                result["prediction"] = pred_values[offset]
                result["ground_truth"] = true_values[offset]
                result["prediction_index_raw"] = float(outputs5[offset].detach().cpu().reshape(-1)[0])
                result["ground_truth_index_raw"] = float(y5[offset].detach().cpu().reshape(-1)[0])
                prediction_rows.append(result)

                sample_metric_row = {column: row[column] for column in FULL_FEATURE_COLUMNS}
                sample_metric_row[FULL_LABEL_COLUMN] = row[FULL_LABEL_COLUMN]
                sample_metric_row[FULL_NAME_COLUMN] = row[FULL_NAME_COLUMN]
                sample_metric_row.update(sample_metrics)
                sample_metric_rows.append(sample_metric_row)

                explainability_row = {column: row[column] for column in FULL_FEATURE_COLUMNS}
                explainability_row[FULL_LABEL_COLUMN] = row[FULL_LABEL_COLUMN]
                explainability_row[FULL_NAME_COLUMN] = row[FULL_NAME_COLUMN]
                explainability_row.update(explainability_stats)
                explainability_rows.append(explainability_row)

            sample_index += x1.size(0)

    all_preds = np.asarray([row["prediction"] for row in prediction_rows], dtype=float)
    all_trues = np.asarray([row["ground_truth"] for row in prediction_rows], dtype=float)
    ss_res = np.sum((all_trues - all_preds) ** 2)
    ss_tot = np.sum((all_trues - all_trues.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    summary_rows = []
    for key, values in metrics.items():
        summary_rows.append({
            "metric": key,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })
    summary_rows.append({"metric": "task5_r2", "mean": float(r2), "std": np.nan, "min": np.nan, "max": np.nan})

    pd.DataFrame(prediction_rows).to_excel(output_dir / f"{prefix}_prediction_results.xlsx", index=False)
    pd.DataFrame(sample_metric_rows).to_excel(output_dir / f"{prefix}_sample_metrics_results.xlsx", index=False)
    pd.DataFrame(explainability_rows).to_excel(output_dir / f"{prefix}_explainability_results.xlsx", index=False)
    pd.DataFrame(summary_rows).to_excel(output_dir / f"{prefix}_metrics_results_summary.xlsx", index=False)
    with (output_dir / f"{prefix}Results.txt").open("w", encoding="utf-8") as f:
        for row in summary_rows:
            f.write(f"{row['metric']}: {row['mean']:.6f}\n")
    print(f"Saved standalone Full R_PMNN results to {output_dir}")


def standalone_main():
    """CLI for independent full-model training/prediction. / 完整模型独立训练与预测入口。"""

    parser = argparse.ArgumentParser(description="Standalone R_PMNN Full training and prediction script.")
    parser.add_argument("--action", default="train,predict", choices=["train", "predict", "train,predict"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--predict-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output-dir", default=str(FULL_OUTPUT_DIR))
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint path; defaults to the selected epoch. / 检查点路径，默认随训练轮数确定。",
    )
    args = parser.parse_args()

    os.chdir(str(CODE_DIR))
    set_random_seed(60)
    configure_full_r_pmnn()
    actions = args.action.split(",")

    _, train_dataset, stats = build_full_dataset(
        FULL_TRAIN_METADATA,
        FULL_TRAIN_IMAGE_ROOT,
        FULL_TRAIN_IMAGE_DIRS,
        fit_stats=True,
    )

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else FULL_CHECKPOINT_DIR / f"{FULL_CHECKPOINT_PREFIX}_epoch_{args.epochs}.pth"
    )
    if "train" in actions:
        epoch_suffix = f"_epoch_{args.epochs}"
        checkpoint_prefix = checkpoint_path.stem
        if checkpoint_prefix.endswith(epoch_suffix):
            checkpoint_prefix = checkpoint_prefix[: -len(epoch_suffix)]
        train_model(
            datasetTrain=train_dataset,
            model_save_dir=str(checkpoint_path.parent),
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            checkpoint_prefix=checkpoint_prefix,
        )
        checkpoint_path = checkpoint_path.parent / f"{checkpoint_prefix}{epoch_suffix}.pth"

    if "predict" in actions:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        predict_frame, predict_dataset, _ = build_full_dataset(
            FULL_PREDICT_METADATA,
            FULL_PREDICT_IMAGE_ROOT,
            FULL_PREDICT_IMAGE_DIRS,
            stats=stats,
            fit_stats=False,
        )
        standalone_predict(
            predict_dataset,
            predict_frame,
            args.output_dir,
            checkpoint_path,
            batch_size=args.predict_batch_size,
        )


if __name__ == '__main__':
    standalone_main()
