import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from resnet18 import ResNet18
from tensorboard_utils import TensorboardLogger

# 自动检测GPU/CPU，优先用CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# 1. 数据加载与预处理
from data_preprocess import get_data_loaders
train_loader, test_loader = get_data_loaders(batch_size=64)

# 2. 定义 CNN 模型
model = ResNet18(in_channels=1, num_classes=10)
model = model.to(device) # 将模型移动到cuda设备

# 3. 定义损失函数与优化器
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 4. 模型训练
num_epochs = 5
model.train()  # 设置模型为训练模式

### TensorBoard功能
logger = TensorboardLogger()
# 添加模型结构到TensorBoard
sample_input = torch.randn(1, 1, 28, 28).to(device)  # MNIST输入是1通道28x28
logger.add_graph(model, sample_input)
# 添加样本图像到TensorBoard
dataiter = iter(train_loader)
images, labels = next(dataiter)
logger.add_images('Sample_Images', images[:16], 0, dataformats='NCHW')
### TensorBoard功能

for epoch in range(num_epochs):
    total_loss = 0
    batch_idx = 0
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
        batch_idx += 1
        
        ### TensorBoard功能
        # 每100个batch记录一次loss
        if batch_idx % 100 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            logger.add_scalar('Training/Loss', loss.item(), global_step)
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    ### TensorBoard功能
    # 每个epoch结束后记录平均loss
    avg_loss = total_loss / len(train_loader)
    logger.add_scalar('Training/Average_Loss', avg_loss, epoch+1)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    # 记录模型参数直方图
    for name, param in model.named_parameters():
        logger.add_histogram(f'Parameters/{name}', param, epoch+1)

# 关闭TensorBoard日志
logger.close()

# 5. 模型测试
model.eval()  # 设置模型为评估模式
correct = 0
total = 0
y_true = []
y_pred = []

with torch.no_grad():  # 关闭梯度计算
    for images, labels in test_loader:
        # 将数据移动到cuda设备
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predict_label = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predict_label == labels).sum().item()
        
        # 收集真实标签和预测标签用于混淆矩阵
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predict_label.cpu().numpy())

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

### TensorBoard功能
# 在TensorBoard中记录测试准确率
logger = TensorboardLogger(log_dir='./logs')
logger.add_scalar('Test/Accuracy', accuracy, num_epochs)
# 添加混淆矩阵
class_names = [str(i) for i in range(10)]  # MNIST数字0-9
logger.add_confusion_matrix('Test/Confusion_Matrix', y_true, y_pred, num_epochs, class_names)
logger.close()
### TensorBoard功能

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
