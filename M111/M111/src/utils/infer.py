import torch
import cv2
import argparse
import numpy as np
from src.model import MambaUNetRainRemoval
from src.utils.utils_helper import pad_img

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="去雨推理")
    parser.add_argument('--image_path', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出图像保存路径')
    parser.add_argument('--pre_trained_model', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--device', type=str, default='cuda', help='运行设备（cuda 或 cpu）')
    return parser.parse_args()

def main():
    args = parse_args()

    # 加载模型
    model = MambaUNetRainRemoval(in_channels=3, out_channels=3, base_c=64)
    model.load_state_dict(torch.load(args.pre_trained_model, map_location=args.device))
    model.to(args.device)
    model.eval()

    # 加载图像
    img = cv2.imread(args.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = pad_img(img_tensor, 32)
    img_tensor = img_tensor.to(args.device)

    # 推理
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.clamp(output, 0, 1)  # 确保输出范围在 [0,1]

    # 保存结果
    output = output.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255.0
    output = cv2.cvtColor(output.astype('uint8'), cv2.COLOR_RGB2BGR)
    # 添加后处理：轻微锐化
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    output = cv2.filter2D(output, -1, kernel)
    cv2.imwrite(args.output_path, output)

if __name__ == "__main__":
    main()