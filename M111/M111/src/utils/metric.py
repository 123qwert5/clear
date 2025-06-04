import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def calculate_psnr(img1, img2, max_val=1.0):
    """
    计算 PSNR
    :param img1: 图像1 (tensor, 范围 [0, 1])
    :param img2: 图像2 (tensor, 范围 [0, 1])
    :param max_val: 最大像素值（默认 1.0）
    :return: PSNR 值
    """
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    psnr_metric = PeakSignalNoiseRatio(data_range=max_val)
    return psnr_metric(img1, img2).item()

def calculate_ssim(img1, img2):
    """
    计算 SSIM
    :param img1: 图像1 (tensor, 范围 [0, 1])
    :param img2: 图像2 (tensor, 范围 [0, 1])
    :return: SSIM 值
    """
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    return ssim_metric(img1, img2).item()

if __name__ == "__main__":
    # 测试指标
    img1 = torch.randn(1, 3, 256, 256)
    img2 = torch.randn(1, 3, 256, 256)
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")