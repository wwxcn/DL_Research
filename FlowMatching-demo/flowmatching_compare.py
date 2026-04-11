"""
Flow Matching (Rectified Flow) 使用 DiT (Diffusion Transformer) 实现
基于论文: "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from dit_model import DiT


class FlowMatching:
    """
    Flow Matching (Rectified Flow) 训练与采样
    """
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None

    def compute_loss(self, x0, x1):
        """
        计算 Flow Matching 损失
        x0: [batch_size, dim] - 噪声样本
        x1: [batch_size, dim] - 数据样本
        """
        batch_size = x0.shape[0]

        # 随机采样时间步 t ~ Uniform(0, 1)
        t = torch.rand(batch_size, device=self.device)

        # 计算插值: x_t = (1 - t) * x0 + t * x1
        t_expanded = t.view(-1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1

        # 真实的向量场 (目标)
        target_v = x1 - x0

        # 模型预测的向量场
        pred_v = self.model(x_t, t)

        # 均方误差损失
        loss = F.mse_loss(pred_v, target_v)

        return loss

    def train_step(self, x1):
        """
        单步训练
        x1: [batch_size, dim] - 数据样本
        """
        batch_size = x1.shape[0]

        # 从先验分布采样噪声
        x0 = torch.randn_like(x1)

        # 计算损失
        loss = self.compute_loss(x0, x1)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, batch_size, num_steps=100):
        """
        使用 ODE 求解器生成样本
        batch_size: 生成的样本数量
        num_steps: ODE 求解的步数
        """
        self.model.eval()

        # 从先验分布采样初始噪声
        x = torch.randn(batch_size, self.model.input_dim, device=self.device)

        dt = 1.0 / num_steps

        # 使用欧拉方法求解 ODE: dx/dt = v(x, t)
        for i in range(num_steps):
            t = torch.ones(batch_size, device=self.device) * (i * dt)

            # 估计向量场
            v = self.model(x, t)

            # 欧拉更新: x_{t+dt} = x_t + dt * v(x_t, t)
            x = x + dt * v

        self.model.train()
        return x

    def train(self, data_loader, num_epochs=200, lr=1e-3):
        """
        训练模型
        """
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        losses = []

        for epoch in range(num_epochs):
            epoch_losses = []

            for batch in data_loader:
                x1 = batch[0].to(self.device)
                loss = self.train_step(x1)
                epoch_losses.append(loss)

            scheduler.step()

            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                lr_current = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, LR: {lr_current:.6f}")

        return losses


def create_2d_dataset(n_samples=20000, dataset_type='swiss_roll'):
    """
    创建2D测试数据集
    """
    if dataset_type == 'swiss_roll':
        # Swiss Roll 数据集
        t = np.random.rand(n_samples) * 4 * np.pi
        x = t * np.cos(t)
        y = t * np.sin(t)
        data = np.stack([x, y], axis=1)
        # 归一化到 [-3, 3] 范围
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        data = data * 2.0

    elif dataset_type == 'moons':
        # 双月数据集
        from sklearn.datasets import make_moons
        data, _ = make_moons(n_samples=n_samples, noise=0.05)
        # 归一化
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        data = data * 2.0

    elif dataset_type == 'gaussian_mixture':
        # 高斯混合
        n_per_cluster = n_samples // 4
        data = []
        for i in range(4):
            angle = i * np.pi / 2
            center = np.array([np.cos(angle), np.sin(angle)]) * 2.5
            cluster = np.random.randn(n_per_cluster, 2) * 0.5 + center
            data.append(cluster)
        data = np.vstack(data)

    return torch.tensor(data, dtype=torch.float32)


def main():
    """
    主函数: 演示使用 DiT 的 Flow Matching 训练和采样
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("\n" + "="*60)
    print("Flow Matching with Diffusion Transformer (DiT)")
    print("="*60)

    # 创建数据集
    print("\nCreating dataset...")
    data = create_2d_dataset(n_samples=20000, dataset_type='swiss_roll')
    print(f"Dataset created with shape: {data.shape}")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"Data mean: {data.mean():.2f}, std: {data.std():.2f}")

    # 创建数据加载器
    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    print(f"DataLoader created with {len(data_loader)} batches")

    # 创建 DiT 模型
    print("\nInitializing DiT model...")
    model = DiT(
        input_dim=2,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 创建 Flow Matching 实例
    flow_matching = FlowMatching(model, device=device)
    print("FlowMatching instance created")

    # 训练
    print("\n" + "="*60)
    print("Training Flow Matching Model")
    print("="*60 + "\n")

    losses = flow_matching.train(data_loader, num_epochs=200, lr=1e-3)
    print(f"\nTraining completed. Final loss: {losses[-1]:.6f}")
    print(f"Loss decreased from {losses[0]:.6f} to {losses[-1]:.6f}")
    print(f"Loss reduction: {100*(losses[0]-losses[-1])/losses[0]:.2f}%")

    # 生成样本
    print("\n" + "="*60)
    print("Generating Samples")
    print("="*60 + "\n")

    generated = flow_matching.sample(1000, num_steps=100).cpu().numpy()
    print(f"Generated {len(generated)} samples")
    print(f"Generated data range: [{generated.min():.2f}, {generated.max():.2f}]")
    print(f"Generated data mean: {generated.mean():.2f}, std: {generated.std():.2f}")

    print(f"\nOriginal data statistics:")
    print(f"  Mean: {data.mean().numpy():.2f}, Std: {data.std().numpy():.2f}")
    print(f"Generated data statistics:")
    print(f"  Mean: {generated.mean():.2f}, Std: {generated.std():.2f}")

    # 测试不同数据集
    print("\n" + "="*60)
    print("Testing with Different Datasets")
    print("="*60 + "\n")

    for dataset_type in ['moons', 'gaussian_mixture']:
        print(f"\nTesting with {dataset_type} dataset...")
        test_data = create_2d_dataset(n_samples=5000, dataset_type=dataset_type)
        test_dataset = TensorDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

        test_model = DiT(
            input_dim=2,
            embed_dim=256,
            depth=6,
            num_heads=8,
            mlp_ratio=4.0
        )
        test_flow = FlowMatching(test_model, device=device)

        print(f"Training on {dataset_type}...")
        test_losses = test_flow.train(test_loader, num_epochs=100, lr=1e-3)

        test_generated = test_flow.sample(500, num_steps=100).cpu().numpy()
        print(f"Final loss: {test_losses[-1]:.6f}")
        print(f"Loss reduction: {100*(test_losses[0]-test_losses[-1])/test_losses[0]:.2f}%")
        print(f"Generated samples: {len(test_generated)}")

    print("\n" + "="*60)
    print("All Tests Completed Successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
