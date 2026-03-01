import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from resnet18 import ResNet18

# 自动检测GPU/CPU，优先用CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# 1. 数据加载与预处理
from data_preprocess import get_data_loaders
train_loader, test_loader = get_data_loaders(batch_size=64)

# 2. 定义 CNN 模型
model = ResNet18(in_channels=1, num_classes=10)
# 将模型移动到cuda设备
model = model.to(device)

# 3. 定义损失函数与优化器
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 4. 模型训练
num_epochs = 5
model.train()  # 设置模型为训练模式

for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        # 将数据移动到cuda设备（核心加速点）
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# 5. 模型测试
model.eval()  # 设置模型为评估模式
correct = 0
total = 0

with torch.no_grad():  # 关闭梯度计算
    for images, labels in test_loader:
        # 将数据移动到cuda设备
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predict_label = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predict_label == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# 保存模型（方便后续加载）
torch.save(model.state_dict(), "resnet18_mnist_cuda.pth")
print("模型已保存为 resnet18_mnist_cuda.pth")

# 6. 可视化测试结果
dataiter = iter(test_loader)
images, labels = next(dataiter)
# 适配模型设备并预测
images_device = images.to(device)
outputs = model(images_device)
_, predictions = torch.max(outputs, 1)
# 转回CPU并反归一化
predictions = predictions.cpu()
labels = labels.cpu()
images = images * 0.5 + 0.5  # 反归一化
# 绘制6张图
fig, axes = plt.subplots(1, 6, figsize=(12, 4))
for i in range(6):
    # 转为numpy数组（H,W）
    img = images[i][0].numpy()
    axes[i].imshow(img, cmap='gray')
    # 显示真实标签和预测标签
    axes[i].set_title(f"True: {labels[i].item()}\nPred: {predictions[i].item()}")
    axes[i].axis('off')
plt.tight_layout()  # 自动调整布局，避免标题重叠
plt.show()
