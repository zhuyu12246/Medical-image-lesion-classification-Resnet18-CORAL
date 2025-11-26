import medmnist
from medmnist import INFO
from torchvision import transforms
#测试下载MNIST数据集
# 任务：序数回归
#
# 标签：
#
# 0: 0，1: 1，2: 2，3: 3，4: 4
#
# 示例：
#
# 列车： 1080
# 验证： 120
# 测试： 400
data_flag = 'retinamnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

# 使用 transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# 指定下载路径 root="./data"
train_dataset = DataClass(
    split='train',
    transform=transform,
    download=True,
    root='./data/RetinaMNIST'
)
val_dataset = DataClass(
    split='val',
    transform=transform,
    download=True,
    root='./data/RetinaMNIST'
)
test_dataset = DataClass(
    split='test',
    transform=transform,
    download=True,
    root='./data/RetinaMNIST'
)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Test size:", len(test_dataset))
