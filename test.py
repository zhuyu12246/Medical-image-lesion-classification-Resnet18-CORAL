import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import RetinaMNIST

from models.resnet18_coral import ResNet18_CORAL

device = "cuda" if torch.cuda.is_available() else "cpu"

# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# dataset
test_dataset = RetinaMNIST(split="test", transform=transform, download=True,root='./data/RetinaMNIST')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
num_classes = int(test_dataset.labels.max() + 1)

# model
model = ResNet18_CORAL(num_classes).to(device)
model.load_state_dict(torch.load("coral_resnet18.pth", map_location=device))
model.eval()

correct = 0
total = 0
#混淆矩阵
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device).squeeze()
        logits = model(x)
        pred = model.predict(logits)
        # 绘制混淆矩阵
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        correct += (pred == y).sum().item()
        total += y.size(0)

acc = correct / total
# print(f"Test Accuracy = {acc:.4f}")
print(f"Test Accuracy = {acc*100:.2f}%")   # 保留两位小数 + %

print("len(all_labels):", len(all_labels))
print("len(all_preds):", len(all_preds))
print("total samples in DataLoader:", len(test_dataset))

# 绘制混淆矩阵
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

num_classes = cm.shape[0]
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)

# 在每个格子里写数字
thresh = cm.max() / 2.
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
# plt.savefig("confusion_matrix.png", dpi=300)
plt.show()