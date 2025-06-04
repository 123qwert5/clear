import torch
import torch.nn as nn

def reparameterize(model):
    """
    重新参数化模型（当前未实现）
    :param model: 模型
    :return: 重新参数化后的模型
    """
    print("重新参数化未实现")
    return model

if __name__ == "__main__":
    from src.model import MambaUNetRainRemoval
    model = MambaUNetRainRemoval()
    reparam_model = reparameterize(model)