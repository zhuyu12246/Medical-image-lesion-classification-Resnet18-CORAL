import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import RetinaMNIST

from models.resnet18_coral import ResNet18_CORAL

device = "cuda" if torch.cuda.is_available() else "cpu"

# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.5], [0.5])
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

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = model.predict(logits)
        correct += (pred == y).sum().item()
        total += len(y)

acc = correct / total
print(f"Test Accuracy = {acc:.4f}")
