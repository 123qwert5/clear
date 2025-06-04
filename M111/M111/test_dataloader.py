import sys
import os
sys.path.insert(0, '/root/M111')
from src.data.data_loader import get_dataloader

try:
    # 修改M111/src/test_dataloader.py
    loader = get_dataloader('/root/autodl-tmp/k12', batch_size=1, mode='training')  # 使用真实路径
    # loader = get_dataloader('/autodl-tmp/k12', batch_size=1, mode='training')
    rain_img, clean_img = next(iter(loader))
    print(rain_img.shape, clean_img.shape)
except Exception as e:
    print(f"错误: {e}")