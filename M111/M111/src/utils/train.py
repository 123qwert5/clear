import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from src.model import MambaUNetRainRemoval
from src.data.data_loader import get_dataloader
from src.loss.cr import CRLoss
from src.utils.option_train import parse_args
from src.utils.eval import evaluate
from src.logger.logger import Logger
from tqdm import tqdm

def train():
    """
    训练 MambaUNetRainRemoval 模型（单视图，RainKITTI2012 数据集，支持两视图）
    """
    args = parse_args()
    logger = Logger(log_dir=args.save_path, log_file='train_log.txt')
    logger.info("开始训练...")

    # 数据集路径
    data_dir = args.data_dir  # 例如 /autodl-tmp/k12

    # 加载训练和测试数据
    logger.info("加载训练数据...")
    train_loader = get_dataloader(
        data_dir=data_dir,
        batch_size=args.batch_size,
        mode='train',
        shuffle=True
    )
    logger.info("加载测试数据...")
    test_loader = get_dataloader(
        data_dir=data_dir,
        batch_size=1,
        mode='test',
        shuffle=False
    )

    logger.info("初始化模型...")
    model = MambaUNetRainRemoval().to(args.device)
    logger.info("模型初始化完成")

    logger.info("初始化优化器和损失函数...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = CRLoss(alpha=0.8).to(args.device)
    logger.info("优化器和损失函数初始化完成")

    os.makedirs(args.save_path, exist_ok=True)

    for epoch in range(args.epochs):
        logger.info(f"开始第 {epoch + 1} 个 epoch...")
        model.train()
        total_loss = 0.0
        for rain_img, clean_img in tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"轮次 {epoch + 1}/{args.epochs}",
            unit="batch"
        ):
            rain_img, clean_img = rain_img.to(args.device), clean_img.to(args.device)
            optimizer.zero_grad()
            output = model(rain_img)
            if output.min() < 0 or output.max() > 1:
                logger.warning(f"输出范围异常: [{output.min().item():.4f}, {output.max().item():.4f}]")
            loss = criterion(output, clean_img)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"轮次 {epoch + 1}/{args.epochs}, 平均损失: {avg_loss:.4f}")

        # 评估
        logger.info("开始评估...")
        model.eval()
        psnr, ssim = evaluate(model, test_loader, args.device)
        logger.info(f"轮次 {epoch + 1}, 测试集 PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

        checkpoint_path = os.path.join(args.save_path, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"检查点已保存: {checkpoint_path}")

if __name__ == "__main__":
    train()