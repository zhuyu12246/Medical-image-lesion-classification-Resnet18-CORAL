import torch
import torch.nn as nn
from torchvision.models import resnet18
# ResNet18 + CORAL


class ResNet18_CORAL(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(512, num_classes - 1)  # CORAL: K-1 outputs

    def forward(self, x):
        return self.model(x)

    def predict(self, logits):
        pred = (logits > 0).sum(dim=1)
        return pred
