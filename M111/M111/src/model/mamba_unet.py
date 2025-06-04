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

# 多尺度特征融合模块 (MSFFM)
class MultiScaleFeatureFusionModule(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(MultiScaleFeatureFusionModule, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])
        self.fusion_conv = nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features, target_size):
        upsampled_features = []
        for i, feat in enumerate(features):
            upsampled = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            upsampled = self.convs[i](upsampled)
            upsampled_features.append(upsampled)
        
        fused = torch.cat(upsampled_features, dim=1)
        fused = self.fusion_conv(fused)
        fused = self.bn(fused)
        fused = self.relu(fused)
        return fused

# 雨纹注意力模块 (RAM)
class RainAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(RainAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        out = self.fc1(avg_out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return x * out

# MambaUNetRainRemoval 模型
class MambaUNetRainRemoval(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_c=64):
        super(MambaUNetRainRemoval, self).__init__()
        self.base_c = base_c
        self.in_channels = in_channels

        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cbam1 = CBAM(base_c)  # 添加 CBAM
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_c, base_c * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c * 2, base_c * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cbam2 = CBAM(base_c * 2)  # 添加 CBAM
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c * 4, base_c * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cbam3 = CBAM(base_c * 4)  # 添加 CBAM
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            nn.Conv2d(base_c * 4, base_c * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c * 8, base_c * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cbam4 = CBAM(base_c * 8)  # 添加 CBAM
        self.pool4 = nn.MaxPool2d(2)

        # 中间层（Mamba 块）
        self.mamba = MambaBlock(base_c * 8, base_c * 8)

        # 多尺度特征融合 (MSFFM)
        self.msffm = MultiScaleFeatureFusionModule(
            in_channels_list=[base_c, base_c * 2, base_c * 4, base_c * 8],
            out_channels=base_c * 8
        )

        # 雨纹注意力模块 (RAM)
        self.ram = RainAttentionModule(in_channels=base_c * 8)

        # 解码器
        self.up4 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=4, stride=4)
        self.dec4 = nn.Sequential(
            nn.Conv2d(base_c * 8, base_c * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c * 4, base_c * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cbam_dec4 = CBAM(base_c * 4)  # 添加 CBAM

        self.up3 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_c * 4, base_c * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c * 2, base_c * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cbam_dec3 = CBAM(base_c * 2)  # 添加 CBAM

        self.up2 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cbam_dec2 = CBAM(base_c)  # 添加 CBAM

        self.up1 = nn.Conv2d(base_c, base_c // 2, kernel_size=1)  # 修改为普通卷积，仅调整通道数
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_c // 2 + in_channels, base_c // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c // 2, base_c // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cbam_dec1 = CBAM(base_c // 2)  # 添加 CBAM

        self.out_conv = nn.Conv2d(base_c // 2, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e1 = self.cbam1(e1)  # 应用 CBAM
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        e2 = self.cbam2(e2)  # 应用 CBAM
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        e3 = self.cbam3(e3)  # 应用 CBAM
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        e4 = self.cbam4(e4)  # 应用 CBAM
        p4 = self.pool4(e4)

        # 中间层（Mamba 块）
        mamba_out = self.mamba(p4)

        # 雨纹注意力模块 (RAM)
        attention_out = self.ram(mamba_out)

        # 多尺度特征融合 (MSFFM)，匹配 attention_out 的尺寸
        fused_features = self.msffm([e1, e2, e3, e4], target_size=attention_out.size()[2:])

        # 融合 Mamba 输出和注意力输出
        mid = fused_features + attention_out

        # 解码器
        d4 = self.up4(mid)
        d4 = torch.cat([d4, e3], dim=1)  # 跳跃连接
        d4 = self.dec4(d4)
        d4 = self.cbam_dec4(d4)  # 应用 CBAM

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)  # 跳跃连接
        d3 = self.dec3(d3)
        d3 = self.cbam_dec3(d3)  # 应用 CBAM

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)  # 跳跃连接
        d2 = self.dec2(d2)
        d2 = self.cbam_dec2(d2)  # 应用 CBAM

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x], dim=1)  # 跳跃连接
        d1 = self.dec1(d1)
        d1 = self.cbam_dec1(d1)  # 应用 CBAM

        out = self.out_conv(d1)
        out = self.sigmoid(out)
        return out