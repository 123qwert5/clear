import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Train MambaUNetRainRemoval on RainKITTI2012")
    parser.add_argument('--data_dir', type=str, default='/autodl-tmp/k12', help='Path to RainKITTI2012 dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Path to save checkpoints')
    return parser.parse_args()