import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def create_window(window_size, channel):
    """
    创建一个高斯窗口，用于 SSIM 计算
    :param window_size: 窗口大小
    :param channel: 通道数
    :return: 高斯窗口张量
    """
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class CustomSSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        """
        自定义 SSIM 损失
        :param window_size: 窗口大小
        :param size_average: 是否平均损失
        """
        super(CustomSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = None

    def forward(self, img1, img2):
        """
        计算 SSIM 损失
        :param img1: 图像1 (batch, channels, height, width)
        :param img2: 图像2 (batch, channels, height, width)
        :return: SSIM 损失值
        """
        # 检查图像范围
        if img1.min() < 0 or img1.max() > 1 or img2.min() < 0 or img2.max() > 1:
            print(f"警告: 图像范围异常 - img1: [{img1.min().item():.4f}, {img1.max().item():.4f}], img2: [{img2.min().item():.4f}, {img2.max().item():.4f}]")
            img1 = torch.clamp(img1, 0, 1)
            img2 = torch.clamp(img2, 0, 1)

        channel = img1.size(1)
        if self.window is None or self.channel != channel:
            self.channel = channel
            self.window = create_window(self.window_size, self.channel)
            if img1.is_cuda:
                self.window = self.window.cuda(img1.get_device())
            self.window = self.window.type_as(img1)

        ssim_value = self._ssim(img1, img2)
        # 转换为损失，确保非负
        ssim_loss = 1 - ssim_value
        return torch.clamp(ssim_loss, min=0)  # 确保损失非负

    def _ssim(self, img1, img2):
        """
        计算 SSIM 值
        """
        padding = int(self.window_size // 2)

        mu1 = F.conv2d(img1, self.window, padding=padding, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=padding, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=padding, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=padding, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=padding, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_map = torch.clamp(ssim_map, 0, 1)  # 确保 SSIM 值在 [0,1] 范围内

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)