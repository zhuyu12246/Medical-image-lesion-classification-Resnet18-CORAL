import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import RetinaMNIST

from models.resnet18_coral import ResNet18_CORAL
from utils.coral import coral_label_transform, coral_loss
# Implementing vision transformer for classifying 2D biomedical images 这篇论文 准确度是57%已经是很高的了
# 本方法准确率大概在57%左右 要想获得最高的效果 需要取得loss最低点的模型
# RetinaMNIST 特点导致容易过拟合
# 数据不均匀 0和2程度的占了大部分了 其他的病变程度 学习到的特征比较少
#
# 数据量小：RetinaMNIST 测试集只有几百张图片，训练集也比较小
#
# 输入图像信息有限：resize 到 224×224，眼底病变细节有限
#
# 网络容量大：ResNet‑18 + CORAL 对小数据集来说，参数量相对大
#
# 结果：模型很快在训练集上拟合，但验证集无法跟上。
device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 32
LR = 1e-5
EPOCHS = 50

# transforms
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((224,224)),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# load datasets
train_dataset = RetinaMNIST(split="train", transform=transform_train, download=True,root='./data/RetinaMNIST')
val_dataset   = RetinaMNIST(split="val",   transform=transform_val, download=True,root='./data/RetinaMNIST')

train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH, shuffle=False)

num_classes = int(train_dataset.labels.max() + 1)

# model
model = ResNet18_CORAL(num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device).long()
        y_coral = coral_label_transform(y, num_classes)

        logits = model(x)
        loss = coral_loss(logits, y_coral)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} | Batch {x.shape[0]} | train_loss={loss.item():.4f}")

    avg_train = total_loss / len(train_loader)
    train_losses.append(avg_train)

    # ---------------- Validate ----------------
    model.eval()
    val_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device).long()
            y_coral = coral_label_transform(y, num_classes)

            logits = model(x)
            val_total += coral_loss(logits, y_coral).item()

    avg_val = val_total / len(val_loader)
    val_losses.append(avg_val)

    print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")

torch.save(train_losses, "train_loss.pt")
torch.save(val_losses, "val_loss.pt")
torch.save(model.state_dict(), "coral_resnet18.pth")

print("训练完成（含验证集）！")
