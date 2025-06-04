import torch
import torch.nn as nn
import torch.nn.functional as F

# CBAM 模块（通道注意力 + 空间注意力）
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # 通道注意力模块
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

        # 空间注意力模块
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # 通道注意力
        avg_pool = self.channel_avg_pool(x).view(x.size(0), -1)
        max_pool = self.channel_max_pool(x).view(x.size(0), -1)
        avg_out = self.channel_fc(avg_pool)
        max_out = self.channel_fc(max_pool)
        channel_attention = self.sigmoid(avg_out + max_out).view(x.size(0), -1, 1, 1)
        x = x * channel_attention

        # 空间注意力
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_attention = self.sigmoid(self.spatial_conv(spatial_input))
        x = x * spatial_attention

        return x

# Mamba 块（占位符）
class MambaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MambaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # 实际的 Mamba 块实现应替换此处

    def forward(self, x):
        return self.conv(x)