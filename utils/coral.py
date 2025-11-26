import torch
import torch.nn.functional as F

# CORAL 工具函数

def coral_label_transform(y, num_classes):
    y = y.view(-1, 1)  # (N, 1)
    out = torch.zeros(y.size(0), num_classes - 1, device=y.device)
    for i in range(num_classes - 1):
        out[:, i] = (y > i).float().squeeze(1)   # 关键：squeeze掉多余的维度
    return out

def coral_loss(logits, y_transformed):
    return F.binary_cross_entropy_with_logits(logits, y_transformed)
