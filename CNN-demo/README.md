# ResNet18 MNIST 手写数字识别项目

基于 PyTorch 实现的 ResNet18 卷积神经网络，用于 MNIST 手写数字分类任务，支持 TensorBoard 可视化训练过程。

## 项目结构

```
CNN-demo/
├── data/                       # MNIST 数据集目录
│   └── MNIST/
│       └── raw/               # 原始数据文件
├── data_preprocess.py         # 数据预处理模块
├── resnet18.py               # ResNet18 模型定义
├── tensorboard_utils.py      # TensorBoard 日志工具
├── train_with_tensorboard.py # 带 TensorBoard 的训练脚本
├── train_without_tensorboard.py # 基础训练脚本
└── analysis.md               # 项目可视化分析文档
```

## 环境要求

- Python 3.7+
- PyTorch 1.9+
- torchvision
- tensorboard
- matplotlib
- scikit-learn
- numpy

## 安装依赖

```bash
pip install torch torchvision tensorboard matplotlib scikit-learn numpy
```

## 快速开始

### 1. 运行训练（带 TensorBoard）

```bash
python train_with_tensorboard.py
```

### 2. 运行训练（不带 TensorBoard）

```bash
python train_without_tensorboard.py
```

## 训练流程

1. **数据加载**: 自动下载 MNIST 数据集（如未下载）
2. **数据预处理**: ToTensor 转换 + 归一化
3. **模型初始化**: ResNet18 (in_channels=1, num_classes=10)
4. **训练循环**: 5 个 epoch，batch_size=64
5. **模型评估**: 在测试集上计算准确率
6. **结果可视化**: 显示预测结果示例

## 模型架构

### ResNet18 结构

```
输入 [N, 1, 28, 28]
    ↓
Stem 层 (Conv7×7 + BN + ReLU + MaxPool)
    ↓ [N, 64, 7, 7]
Layer1 (2×BasicBlock1)
    ↓ [N, 64, 7, 7]
Layer2 (1×BasicBlock2 + 1×BasicBlock1)
    ↓ [N, 128, 4, 4]
Layer3 (1×BasicBlock2 + 1×BasicBlock1)
    ↓ [N, 256, 2, 2]
Layer4 (1×BasicBlock2 + 1×BasicBlock1)
    ↓ [N, 512, 1, 1]
Head 层 (AvgPool + Flatten + Linear)
    ↓ [N, 10]
输出 Logits
```

### 残差块类型

- **BasicBlock1**: 通道/尺寸不变的残差块
- **BasicBlock2**: 通道翻倍、尺寸减半的下采样残差块

## 超参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 64 | 批次大小 |
| num_epochs | 5 | 训练轮数 |
| learning_rate | 0.01 | 学习率 |
| momentum | 0.9 | SGD 动量系数 |
| optimizer | SGD | 优化器 |
| criterion | CrossEntropyLoss | 损失函数 |

## TensorBoard 可视化

### 启动 TensorBoard

```bash
tensorboard --logdir=./logs
```

然后在浏览器中打开 `http://localhost:6006`

### 监控指标

- **Training/Loss**: 每 100 批次记录的训练损失
- **Training/Average_Loss**: 每个 epoch 的平均损失
- **Parameters/***: 各层参数的直方图分布
- **Sample_Images**: 训练样本可视化
- **Test/Accuracy**: 测试准确率
- **Test/Confusion_Matrix**: 混淆矩阵

## 输出文件

| 文件 | 说明 |
|------|------|
| `resnet18_mnist_cuda.pth` | 训练好的模型权重 |
| `./logs/train_YYYYMMDD_HHMMSS/` | TensorBoard 日志目录 |
| `./data/MNIST/` | 下载的 MNIST 数据集 |

## 详细分析文档

查看 [analysis.md](./analysis.md) 获取完整的 Mermaid 可视化流程图，包括：

1. 整体训练流程可视化图
2. 模型 Forward 数据流向可视化图
3. 损失计算流程可视化图
4. 模型结构总体可视化图
5. 模型结构细节可视化图

## 代码说明

### data_preprocess.py

数据加载和预处理模块，提供 `get_data_loaders()` 函数：

- 自动检测 MNIST 是否已下载
- 应用 ToTensor 和 Normalize 变换
- 创建 DataLoader（支持 pin_memory 加速 GPU 传输）

### resnet18.py

ResNet18 模型定义，包含：

- `BasicBlock1`: 恒等残差块
- `BasicBlock2`: 下采样残差块
- `ResNet18`: 主模型类

### tensorboard_utils.py

TensorBoard 日志工具类 `TensorboardLogger`，提供：

- `add_scalar`: 添加标量数据
- `add_histogram`: 添加参数直方图
- `add_images`: 添加图像数据
- `add_graph`: 添加模型结构图
- `add_confusion_matrix`: 添加混淆矩阵

### train_with_tensorboard.py

完整的训练脚本，包含：

- 设备自动检测（CUDA/CPU）
- 模型训练和评估
- TensorBoard 日志记录
- 模型保存和结果可视化

## 性能优化

- **GPU 加速**: 自动检测并使用 CUDA
- **pin_memory**: 加速 CPU 到 GPU 的数据传输
- **inplace ReLU**: 节省内存
- **no_grad 评估**: 测试时禁用梯度计算

## 许可证

MIT License
