# Flow Matching with Diffusion Transformer (DiT)

基于 Flow Matching (Rectified Flow) 和 Diffusion Transformer (DiT) 的生成模型实现。

## 项目概述

本项目实现了一个经典的 Flow Matching 生成模型，使用 DiT (Diffusion Transformer) 作为向量场估计网络。通过学习从噪声分布到数据分布的直线路径，实现高效的样本生成。项目包含完整的实现、演示代码和用户实现文件。

## 核心组件

| 文件 | 功能描述 |
|------|---------|
| `flow_matching_dit_model.py` | DiT 模型实现，包含多头注意力、MLP、Transformer 块 |
| `flow_matching_class.py` | Flow Matching 核心逻辑：损失计算、训练、采样 |
| `flow_matching_main.py` | 主程序：数据集创建、模型训练、样本生成演示 |

## 模型架构

### DiT (Diffusion Transformer)

```
DiT(
  input_dim: 输入维度
  embed_dim: 256 (嵌入维度)
  depth: 6 (Transformer 层数)
  num_heads: 8 (注意力头数)
  mlp_ratio: 4.0 (MLP 扩展比例)
)
```

主要组件：
- **PatchEmbedding**: 将输入数据投影到高维空间
- **TimeEmbedding**: 正弦位置编码的时间步嵌入
- **DiTBlock**: 包含 LayerNorm、自注意力和条件注入的 Transformer 块
- **MultiHeadAttention**: 多头自注意力机制

### Flow Matching

- **损失函数**: $L = \mathbb{E}_{t,x_0,x_1}[\|v_\theta(x_t,t) - (x_1-x_0)\|^2]$
- **采样**: 欧拉方法求解 ODE $\frac{dx}{dt} = v_\theta(x, t)$

## 算法原理

### Flow Matching 核心公式

| 概念 | 公式 |
|------|------|
| 插值路径 | $x_t = (1-t)x_0 + tx_1$ |
| 目标向量场 | $u_t = x_1 - x_0$ |
| 概率流 ODE | $\frac{dx}{dt} = v_\theta(x, t)$ |

### 训练流程

1. 从数据分布采样 $x_1$
2. 从先验分布采样噪声 $x_0 \sim \mathcal{N}(0, I)$
3. 随机采样时间步 $t \sim \text{Uniform}(0, 1)$
4. 计算插值 $x_t = (1-t)x_0 + tx_1$
5. 最小化 $\|v_\theta(x_t, t) - (x_1 - x_0)\|^2$

### 采样流程

1. 从先验分布采样初始噪声 $x_0$
2. 使用欧拉方法迭代更新：
   - $t_i = i \cdot \Delta t$
   - $v = v_\theta(x_i, t_i)$
   - $x_{i+1} = x_i + \Delta t \cdot v$

## 安装依赖

```bash
pip install torch numpy matplotlib scikit-learn
```

## 使用方法

### 运行演示

```bash
python flow_matching_main.py
```

### 基本用法

```python
from flow_matching_dit_model import DiT
from flow_matching_class import FlowMatching
from torch.utils.data import DataLoader, TensorDataset
import torch

# 创建模型
model = DiT(input_dim=2, embed_dim=256, depth=6, num_heads=8)

# 创建 Flow Matching 实例
flow_matching = FlowMatching(model, device='cuda')

# 准备数据
data = your_data_tensor  # [N, 2]
dataset = TensorDataset(data)
data_loader = DataLoader(dataset, batch_size=512, shuffle=True)

# 训练
losses = flow_matching.train(data_loader, num_epochs=200, lr=1e-3)

# 生成样本
samples = flow_matching.sample(batch_size=1000, num_steps=100)
```

## 支持的数据集

- **Swiss Roll**: 螺旋形 3D 数据投影到 2D
- **Moons**: sklearn 双月形数据
- **Gaussian Mixture**: 4 簇高斯混合分布

## 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 512 | 批大小 |
| num_epochs | 200 | 训练轮数 |
| learning_rate | 1e-3 | 学习率 |
| weight_decay | 0.01 | AdamW 权重衰减 |
| num_steps | 100 | 采样步数 |

## 参考论文

- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)

## 项目结构

```
FlowMatching-demo/
├── flow_matching_dit_model.py            # DiT 模型实现
├── flow_matching_class.py          # Flow Matching 核心逻辑
├── flow_matching_main.py     # 主程序演示
├── flow_matching_dit+class+main_wwx.py   # 用户实现的文件，包含上面三个文件的功能，面试时使用
└── README.md                 # 本文档
```

## 文件说明

### 主要实现文件

| 文件 | 用途 |
|------|------|
| `dit_model.py` | 完整的 DiT (Diffusion Transformer) 模型实现，可用于任何需要条件生成的场景 |
| `flow_matching.py` | Flow Matching 的训练、损失计算和采样逻辑，是一个通用的包装器 |
| `flow_matching_demo.py` | 使用 DiT 和 Flow Matching 在 2D 数据集上的完整演示 |
