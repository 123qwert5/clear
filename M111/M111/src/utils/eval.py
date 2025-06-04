import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def evaluate(model, dataloader, device):
    """
    评估模型（训练和测试集都有干净图像）
    """
    model.eval()
    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for rain_img, clean_img in dataloader:
            rain_img, clean_img = rain_img.to(device), clean_img.to(device)
            output = model(rain_img)
            output = torch.clamp(output, 0, 1)

            # 计算 PSNR 和 SSIM
            output_np = output.cpu().numpy().transpose(0, 2, 3, 1)
            clean_np = clean_img.cpu().numpy().transpose(0, 2, 3, 1)
            for i in range(output_np.shape[0]):
                psnr_val = psnr(clean_np[i], output_np[i], data_range=1.0)
                ssim_val = ssim(clean_np[i], output_np[i], multichannel=True, data_range=1.0, channel_axis=2)
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    return avg_psnr, avg_ssim