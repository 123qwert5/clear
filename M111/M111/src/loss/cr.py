import torch
import torch.nn as nn
from .custom_ssim import CustomSSIM  # 导入自定义的 CustomSSIM

class CRLoss(nn.Module):
    def __init__(self, alpha=0.8):
        """
        初始化组合损失（L1 + SSIM）
        :param alpha: L1 损失的权重（0 到 1 之间）
        """
        super(CRLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = CustomSSIM(window_size=11, size_average=True)

    def forward(self, pred, target):
        """
        计算组合损失
        :param pred: 模型预测图像 (batch, channels, height, width)
        :param target: 真实干净图像 (batch, channels, height, width)
        :return: 组合损失值
        """
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)  # SSIM 损失已经包含 1 - SSIM
        total_loss = self.alpha * l1 + (1 - self.alpha) * ssim
        return total_loss