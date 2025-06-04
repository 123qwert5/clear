import torch

def pad_img(img, size):
    """
    将图像尺寸填充到指定大小的倍数
    :param img: 张量 (batch, channels, height, width)
    :param size: 目标尺寸的倍数
    :return: 填充后的张量
    """
    b, c, h, w = img.shape
    pad_h = (size - h % size) % size
    pad_w = (size - w % size) % size
    if pad_h > 0 or pad_w > 0:
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return img