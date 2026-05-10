# ResNet18 MNIST 训练项目可视化分析

## 1. 整体训练流程可视化图

```mermaid
flowchart TD
    subgraph "数据准备阶段"
        A["开始"] --> B["MNIST数据集加载<br/>datasets.MNIST"]
        B --> C["数据预处理<br/>ToTensor + Normalize"]
        C --> D["创建DataLoader<br/>batch_size=64"]
    end

    subgraph "模型初始化阶段"
        D --> E["设备检测<br/>CUDA/CPU"]
        E --> F["实例化ResNet18<br/>in_channels=1, num_classes=10"]
        F --> G["模型移至设备<br/>model.to(device)"]
        G --> H["定义损失函数<br/>CrossEntropyLoss"]
        H --> I["定义优化器<br/>SGD lr=0.01 momentum=0.9"]
    end

    subgraph "TensorBoard初始化"
        I --> J["创建TensorboardLogger"]
        J --> K["添加模型结构<br/>add_graph"]
        K --> L["添加样本图像<br/>add_images"]
    end

    subgraph "训练循环<br/>num_epochs=5"
        L --> M["设置训练模式<br/>model.train"]
        M --> N["遍历train_loader"]
        N --> O["数据移至device"]
        O --> P["optimizer.zero_grad"]
        P --> Q["前向传播<br/>model(images)"]
        Q --> R["计算损失<br/>criterion(outputs, labels)"]
        R --> S["反向传播<br/>loss.backward"]
        S --> T["参数更新<br/>optimizer.step"]
        T --> U{"batch_idx % 100 == 0?"}
        U -->|"是"| V["记录Loss到TensorBoard"]
        U -->|"否"| W{"还有批次?"}
        V --> W
        W -->|"是"| N
        W -->|"否"| X["记录平均Loss"]
        X --> Y["记录参数直方图"]
        Y --> Z{"还有epoch?"}
        Z -->|"是"| M
        Z -->|"否"| AA["关闭TensorBoard日志"]
    end

    subgraph "测试评估阶段"
        AA --> AB["设置评估模式<br/>model.eval"]
        AB --> AC["遍历test_loader"]
        AC --> AD["数据移至device"]
        AD --> AE["前向传播<br/>outputs = model(images)"]
        AE --> AF["获取预测结果<br/>torch.max"]
        AF --> AG["统计正确数"]
        AG --> AH{"还有批次?"}
        AH -->|"是"| AC
        AH -->|"否"| AI["计算准确率"]
        AI --> AJ["记录测试准确率"]
        AJ --> AK["生成混淆矩阵"]
    end

    subgraph "结果保存与可视化"
        AK --> AL["保存模型<br/>resnet18_mnist_cuda.pth"]
        AL --> AM["可视化预测结果<br/>matplotlib"]
        AM --> AN["结束"]
    end
```

### 关键点说明

| 阶段 | 关键操作 | 说明 |
|------|----------|------|
| 数据准备 | `ToTensor + Normalize((0.5,), (0.5,))` | 将图像归一化到[-1, 1]范围 |
| 模型初始化 | `ResNet18(in_channels=1, num_classes=10)` | 适配MNIST单通道灰度图像 |
| 优化器 | `SGD(lr=0.01, momentum=0.9)` | 带动量的随机梯度下降 |
| 训练循环 | `batch_size=64, num_epochs=5` | 每100批次记录一次loss |
| 设备管理 | `images.to(device), labels.to(device)` | 数据与模型必须在同一设备 |
| TensorBoard | `add_scalar, add_histogram, add_confusion_matrix` | 多维度训练监控 |

---

## 2. 模型Forward数据流向可视化图

```mermaid
flowchart LR
    subgraph "输入层"
        A["输入图像<br/>[N, 1, 28, 28]"] --> B["Conv2d<br/>kernel=7, stride=2, pad=3<br/>[N, 64, 14, 14]"]
        B --> C["BatchNorm2d<br/>[N, 64, 14, 14]"]
        C --> D["ReLU<br/>[N, 64, 14, 14]"]
        D --> E["MaxPool2d<br/>kernel=3, stride=2, pad=1<br/>[N, 64, 7, 7]"]
    end

    subgraph "Layer1<br/>2×BasicBlock1"
        E --> F["BasicBlock1-1<br/>[N, 64, 7, 7] → [N, 64, 7, 7]"]
        F --> G["BasicBlock1-2<br/>[N, 64, 7, 7] → [N, 64, 7, 7]"]
    end

    subgraph "Layer2<br/>1×BasicBlock2 + 1×BasicBlock1"
        G --> H["BasicBlock2<br/>[N, 64, 7, 7] → [N, 128, 4, 4]"]
        H --> I["BasicBlock1<br/>[N, 128, 4, 4] → [N, 128, 4, 4]"]
    end

    subgraph "Layer3<br/>1×BasicBlock2 + 1×BasicBlock1"
        I --> J["BasicBlock2<br/>[N, 128, 4, 4] → [N, 256, 2, 2]"]
        J --> K["BasicBlock1<br/>[N, 256, 2, 2] → [N, 256, 2, 2]"]
    end

    subgraph "Layer4<br/>1×BasicBlock2 + 1×BasicBlock1"
        K --> L["BasicBlock2<br/>[N, 256, 2, 2] → [N, 512, 1, 1]"]
        L --> M["BasicBlock1<br/>[N, 512, 1, 1] → [N, 512, 1, 1]"]
    end

    subgraph "输出层"
        M --> N["AdaptiveAvgPool2d<br/>[N, 512, 1, 1]"]
        N --> O["Flatten<br/>[N, 512]"]
        O --> P["Linear<br/>[N, 512] → [N, 10]"]
        P --> Q["输出Logits<br/>[N, 10]"]
    end
```

### 关键点说明

| 层级 | 输入维度 | 输出维度 | 操作说明 |
|------|----------|----------|----------|
| 输入层 | [N, 1, 28, 28] | [N, 64, 7, 7] | 7×7卷积(stride=2) + 3×3最大池化(stride=2) |
| Layer1 | [N, 64, 7, 7] | [N, 64, 7, 7] | 2个通道不变的残差块 |
| Layer2 | [N, 64, 7, 7] | [N, 128, 4, 4] | 1个下采样块(通道翻倍、尺寸减半) + 1个不变块 |
| Layer3 | [N, 128, 4, 4] | [N, 256, 2, 2] | 同上，继续下采样 |
| Layer4 | [N, 256, 2, 2] | [N, 512, 1, 1] | 同上，最终特征图1×1 |
| 输出层 | [N, 512, 1, 1] | [N, 10] | 全局平均池化 + 全连接分类 |

**N = batch_size**

---

## 3. 损失计算流程可视化图

```mermaid
flowchart TD
    subgraph "前向传播输出"
        A["模型输出<br/>outputs<br/>[N, 10]"] --> B["Softmax<br/>将logits转为概率<br/>[N, 10]"]
    end

    subgraph "标签处理"
        C["真实标签<br/>labels<br/>[N]"] --> D["One-Hot编码<br/>[N, 10]"]
    end

    subgraph "CrossEntropyLoss计算"
        B --> E["计算负对数似然<br/>-log(p[true_class])"]
        D --> E
        E --> F["求平均<br/>loss = mean(-log(p))"]
    end

    subgraph "反向传播"
        F --> G["loss.backward<br/>计算梯度"]
        G --> H["梯度传播至<br/>所有可训练参数"]
    end

    subgraph "参数更新"
        H --> I["optimizer.step<br/>更新权重"]
        I --> J["w_new = w_old - lr * grad"]
    end
```

### 关键点说明

| 步骤 | 操作 | 数学表达 | 说明 |
|------|------|----------|------|
| 模型输出 | `outputs = model(images)` | $z \in \mathbb{R}^{N \times 10}$ | 未经归一化的logits |
| Softmax | $\sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | $p \in (0, 1)^{N \times 10}$ | 转为概率分布 |
| 交叉熵 | $-\sum y_i \log(p_i)$ | $loss \in \mathbb{R}$ | 仅计算真实类别的负对数概率 |
| 反向传播 | `loss.backward()` | $\frac{\partial loss}{\partial w}$ | 自动计算所有参数的梯度 |
| 参数更新 | `optimizer.step()` | $w := w - \eta \nabla_w$ | SGD带动量更新参数 |

**注意**: PyTorch的`nn.CrossEntropyLoss()`内部已经包含了Softmax操作，所以模型输出不需要额外经过Softmax。

---

## 4. 模型结构总体可视化图

```mermaid
flowchart TD
    subgraph "ResNet18整体架构"
        A["输入<br/>[N, 1, 28, 28]"] --> B["Stem层"]
        B --> C["Layer1"]
        C --> D["Layer2"]
        D --> E["Layer3"]
        E --> F["Layer4"]
        F --> G["Head层"]
        G --> H["输出<br/>[N, 10]"]
    end

    subgraph "Stem层结构"
        B --> B1["Conv2d<br/>1→64, 7×7, s=2"]
        B1 --> B2["BatchNorm2d<br/>64"]
        B2 --> B3["ReLU"]
        B3 --> B4["MaxPool2d<br/>3×3, s=2"]
    end

    subgraph "Layer1结构<br/>尺寸不变"
        C --> C1["BasicBlock1<br/>64→64"]
        C1 --> C2["BasicBlock1<br/>64→64"]
    end

    subgraph "Layer2结构<br/>尺寸减半"
        D --> D1["BasicBlock2<br/>64→128"]
        D1 --> D2["BasicBlock1<br/>128→128"]
    end

    subgraph "Layer3结构<br/>尺寸减半"
        E --> E1["BasicBlock2<br/>128→256"]
        E1 --> E2["BasicBlock1<br/>256→256"]
    end

    subgraph "Layer4结构<br/>尺寸减半"
        F --> F1["BasicBlock2<br/>256→512"]
        F1 --> F2["BasicBlock1<br/>512→512"]
    end

    subgraph "Head层结构"
        G --> G1["AdaptiveAvgPool2d<br/>→1×1"]
        G1 --> G2["Flatten"]
        G2 --> G3["Linear<br/>512→10"]
    end
```

### 关键点说明

| 模块 | 组成 | 输入通道 | 输出通道 | 特征图尺寸 | 参数量级 |
|------|------|----------|----------|------------|----------|
| Stem层 | Conv + BN + ReLU + MaxPool | 1 | 64 | 28×28 → 7×7 | ~3K |
| Layer1 | 2×BasicBlock1 | 64 | 64 | 7×7 | ~73K |
| Layer2 | 1×BasicBlock2 + 1×BasicBlock1 | 64 | 128 | 7×7 → 4×4 | ~230K |
| Layer3 | 1×BasicBlock2 + 1×BasicBlock1 | 128 | 256 | 4×4 → 2×2 | ~919K |
| Layer4 | 1×BasicBlock2 + 1×BasicBlock1 | 256 | 512 | 2×2 → 1×1 | ~3.6M |
| Head层 | AvgPool + Flatten + Linear | 512 | 10 | 1×1 → 10 | ~5K |

**总参数量**: 约 11M (MNIST适配版ResNet18)

---

## 5. 模型结构细节可视化图

### 5.1 BasicBlock1 结构细节（通道/尺寸不变）

```mermaid
flowchart TD
    subgraph "BasicBlock1<br/>输入输出维度相同"
        A["输入 x_input<br/>[N, C, H, W]"] --> B["主分支"]
        A --> C["残差连接<br/>Shortcut"]

        subgraph "主分支"
            B --> B1["Conv2d<br/>3×3, s=1, p=1<br/>C→C"]
            B1 --> B2["BatchNorm2d<br/>C"]
            B2 --> B3["ReLU"]
            B3 --> B4["Conv2d<br/>3×3, s=1, p=1<br/>C→C"]
            B4 --> B5["BatchNorm2d<br/>C"]
        end

        C --> C1["恒等映射<br/>x_input"]

        B5 --> D["逐元素相加<br/>+"]
        C1 --> D
        D --> E["ReLU"]
        E --> F["输出<br/>[N, C, H, W]"]
    end
```

### 5.2 BasicBlock2 结构细节（通道翻倍/尺寸减半）

```mermaid
flowchart TD
    subgraph "BasicBlock2<br/>下采样残差块"
        A["输入 x_input<br/>[N, C, H, W]"] --> B["主分支"]
        A --> C["Shortcut分支"]

        subgraph "主分支"
            B --> B1["Conv2d<br/>3×3, s=2, p=1<br/>C→2C"]
            B1 --> B2["BatchNorm2d<br/>2C"]
            B2 --> B3["ReLU"]
            B3 --> B4["Conv2d<br/>3×3, s=1, p=1<br/>2C→2C"]
            B4 --> B5["BatchNorm2d<br/>2C"]
        end

        subgraph "Shortcut分支"
            C --> C1["Conv2d<br/>1×1, s=2, p=0<br/>C→2C"]
            C1 --> C2["BatchNorm2d<br/>2C"]
        end

        B5 --> D["逐元素相加<br/>+"]
        C2 --> D
        D --> E["ReLU"]
        E --> F["输出<br/>[N, 2C, H/2, W/2]"]
    end
```

### 5.3 各层维度变化详细表

```mermaid
flowchart LR
    subgraph "MNIST输入处理"
        A1["原始图像<br/>28×28×1"] --> A2["Conv7×7,s2<br/>14×14×64"]
        A2 --> A3["MaxPool3×3,s2<br/>7×7×64"]
    end

    subgraph "Layer1<br/>2个BasicBlock1"
        A3 --> B1["Block1-1<br/>7×7×64"] --> B2["Block1-2<br/>7×7×64"]
    end

    subgraph "Layer2<br/>下采样"
        B2 --> C1["Block2-1<br/>4×4×128"] --> C2["Block2-2<br/>4×4×128"]
    end

    subgraph "Layer3<br/>下采样"
        C2 --> D1["Block3-1<br/>2×2×256"] --> D2["Block3-2<br/>2×2×256"]
    end

    subgraph "Layer4<br/>下采样"
        D2 --> E1["Block4-1<br/>1×1×512"] --> E2["Block4-2<br/>1×1×512"]
    end

    subgraph "分类头"
        E2 --> F1["AvgPool<br/>1×1×512"] --> F2["Flatten<br/>512"] --> F3["FC<br/>10"]
    end
```

### 关键点说明

#### BasicBlock1（恒等残差块）

| 组件 | 配置 | 维度变化 | 作用 |
|------|------|----------|------|
| Conv1 | 3×3, stride=1, padding=1 | [N,C,H,W]→[N,C,H,W] | 特征提取 |
| BN1 | num_features=C | [N,C,H,W] | 批归一化 |
| ReLU | inplace=True | [N,C,H,W] | 非线性激活 |
| Conv2 | 3×3, stride=1, padding=1 | [N,C,H,W]→[N,C,H,W] | 特征提取 |
| BN2 | num_features=C | [N,C,H,W] | 批归一化 |
| Shortcut | 恒等映射 | [N,C,H,W] | 残差连接 |

#### BasicBlock2（下采样残差块）

| 组件 | 配置 | 维度变化 | 作用 |
|------|------|----------|------|
| Conv0 (Shortcut) | 1×1, stride=2, padding=0 | [N,C,H,W]→[N,2C,H/2,W/2] | 调整shortcut维度 |
| BN0 | num_features=2C | [N,2C,H/2,W/2] | shortcut批归一化 |
| Conv1 | 3×3, stride=2, padding=1 | [N,C,H,W]→[N,2C,H/2,W/2] | 下采样+通道翻倍 |
| BN1 | num_features=2C | [N,2C,H/2,W/2] | 批归一化 |
| ReLU | inplace=True | [N,2C,H/2,W/2] | 非线性激活 |
| Conv2 | 3×3, stride=1, padding=1 | [N,2C,H/2,W/2]→[N,2C,H/2,W/2] | 特征提取 |
| BN2 | num_features=2C | [N,2C,H/2,W/2] | 批归一化 |

#### 残差连接的作用

1. **解决梯度消失**: 通过跳跃连接，梯度可以直接反向传播到浅层
2. **恒等映射**: 学习残差映射 $F(x) = H(x) - x$ 比直接学习 $H(x)$ 更容易
3. **网络深度**: 允许训练非常深的网络（ResNet可以有100+层）

---

## 6. 其他补充说明点

### 6.1 TensorBoard监控指标

```mermaid
flowchart LR
    subgraph "TensorBoard可视化内容"
        A["标量指标"] --> A1["Training/Loss<br/>每100批次"]
        A --> A2["Training/Average_Loss<br/>每epoch"]
        A --> A3["Test/Accuracy<br/>训练结束后"]

        B["图像"] --> B1["Sample_Images<br/>训练样本可视化"]
        B --> B2["Confusion_Matrix<br/>测试混淆矩阵"]

        C["直方图"] --> C1["Parameters/*<br/>各层参数分布"]

        D["计算图"] --> D1["模型结构图<br/>add_graph"]
    end
```

### 6.2 数据流设备管理

```mermaid
flowchart TD
    subgraph "CPU内存"
        A["MNIST数据集<br/>磁盘加载"] --> B["DataLoader<br/>batch生成"]
    end

    subgraph "GPU显存"
        C["模型参数<br/>ResNet18"] --> D["前向传播计算"]
        D --> E["损失计算"]
        E --> F["反向传播"]
        F --> G["参数更新"]
    end

    B --> |"images.to(device)"| H["数据传输"]
    H --> D
```

### 6.3 关键超参数汇总

| 超参数 | 值 | 说明 |
|--------|-----|------|
| batch_size | 64 | 每批次样本数 |
| num_epochs | 5 | 训练轮数 |
| learning_rate | 0.01 | 学习率 |
| momentum | 0.9 | SGD动量系数 |
| optimizer | SGD | 优化器类型 |
| criterion | CrossEntropyLoss | 损失函数 |
| num_workers | 0 | 数据加载线程数 |
| pin_memory | True | 锁页内存加速GPU传输 |

### 6.4 文件依赖关系

```mermaid
flowchart TD
    A["train_with_tersorboard.py<br/>主训练脚本"] --> B["resnet18.py<br/>模型定义"]
    A --> C["data_preprocess.py<br/>数据预处理"]
    A --> D["tensorboard_utils.py<br/>日志工具"]

    B --> B1["BasicBlock1<br/>恒等残差块"]
    B --> B2["BasicBlock2<br/>下采样残差块"]
    B --> B3["ResNet18<br/>主模型类"]

    C --> C1["get_data_loaders<br/>数据加载函数"]

    D --> D1["TensorboardLogger<br/>日志记录类"]
```

### 6.5 运行命令

```bash
python train_with_tersorboard.py
```

### 6.6 输出文件

| 文件 | 说明 |
|------|------|
| `resnet18_mnist_cuda.pth` | 训练好的模型权重 |
| `./logs/train_YYYYMMDD_HHMMSS/` | TensorBoard日志目录 |
| `./data/MNIST/` | 下载的MNIST数据集 |

### 6.7 查看TensorBoard

```bash
tensorboard --logdir=./logs
```

然后在浏览器中打开 `http://localhost:6006` 查看训练可视化。
