"""
测试 Flow Matching 实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


class TimeEmbedding(nn.Module):
    """时间步嵌入模块"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class UNet1D(nn.Module):
    """1D U-Net 用于估计向量场"""
    def __init__(self, input_dim, hidden_dim=256, time_emb_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.time_emb_dim = time_emb_dim

        self.time_embed = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h1 = self.encoder1(torch.cat([x, t_emb], dim=-1))
        h2 = self.encoder2(h1)
        h = self.bottleneck(h2)
        h = self.decoder2(torch.cat([h, h2], dim=-1))
        h = self.decoder1(torch.cat([h, h1], dim=-1))
        return h


class FlowMatching:
    """Flow Matching 训练与采样"""
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None

    def compute_loss(self, x0, x1):
        batch_size = x0.shape[0]
        t = torch.rand(batch_size, device=self.device)
        t_expanded = t.view(-1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        target_v = x1 - x0
        pred_v = self.model(x_t, t)
        loss = F.mse_loss(pred_v, target_v)
        return loss

    def train_step(self, x1):
        batch_size = x1.shape[0]
        x0 = torch.randn_like(x1)
        loss = self.compute_loss(x0, x1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, batch_size, num_steps=100, return_trajectory=False):
        self.model.eval()
        x = torch.randn(batch_size, self.model.input_dim, device=self.device)
        trajectory = [x.cpu().numpy()] if return_trajectory else None
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.ones(batch_size, device=self.device) * (i * dt)
            v = self.model(x, t)
            x = x + dt * v
            if return_trajectory:
                trajectory.append(x.cpu().numpy())

        self.model.train()

        if return_trajectory:
            return x, np.array(trajectory)
        return x

    def train(self, data_loader, num_epochs=200, lr=1e-3):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
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
    """创建2D测试数据集"""
    if dataset_type == 'swiss_roll':
        t = np.random.rand(n_samples) * 4 * np.pi
        x = t * np.cos(t)
        y = t * np.sin(t)
        data = np.stack([x, y], axis=1)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        data = data * 2.0

    elif dataset_type == 'moons':
        from sklearn.datasets import make_moons
        data, _ = make_moons(n_samples=n_samples, noise=0.05)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        data = data * 2.0

    elif dataset_type == 'gaussian_mixture':
        n_per_cluster = n_samples // 4
        data = []
        for i in range(4):
            angle = i * np.pi / 2
            center = np.array([np.cos(angle), np.sin(angle)]) * 2.5
            cluster = np.random.randn(n_per_cluster, 2) * 0.5 + center
            data.append(cluster)
        data = np.vstack(data)

    return torch.tensor(data, dtype=torch.float32)


def visualize_results(flow_matching, data, save_path='results.png'):
    """可视化训练结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].scatter(data[:, 0], data[:, 1], alpha=0.5, s=1)
    axes[0, 0].set_title('Original Data')
    axes[0, 0].set_aspect('equal')

    prior = torch.randn(1000, 2)
    axes[0, 1].scatter(prior[:, 0], prior[:, 1], alpha=0.5, s=1)
    axes[0, 1].set_title('Prior Distribution (Gaussian)')
    axes[0, 1].set_aspect('equal')

    generated = flow_matching.sample(1000, num_steps=100).cpu().numpy()
    axes[0, 2].scatter(generated[:, 0], generated[:, 1], alpha=0.5, s=1)
    axes[0, 2].set_title('Generated Samples')
    axes[0, 2].set_aspect('equal')

    _, trajectory = flow_matching.sample(100, num_steps=50, return_trajectory=True)

    for i in range(min(20, trajectory.shape[1])):
        axes[1, 0].plot(trajectory[:, i, 0], trajectory[:, i, 1], alpha=0.3, linewidth=0.5)
    axes[1, 0].scatter(trajectory[0, :20, 0], trajectory[0, :20, 1], c='blue', s=10, label='Start')
    axes[1, 0].scatter(trajectory[-1, :20, 0], trajectory[-1, :20, 1], c='red', s=10, label='End')
    axes[1, 0].set_title('Generation Trajectories')
    axes[1, 0].legend()
    axes[1, 0].set_aspect('equal')

    x_grid = np.linspace(-3, 3, 20)
    y_grid = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32).to(flow_matching.device)

    for idx, t_val in enumerate([0.0, 0.5, 1.0]):
        t = torch.ones(grid_points.shape[0], device=flow_matching.device) * t_val
        with torch.no_grad():
            v = flow_matching.model(grid_points, t).cpu().numpy()

        U = v[:, 0].reshape(X.shape)
        V = v[:, 1].reshape(X.shape)

        ax = axes[1, 1] if idx < 2 else axes[1, 2]
        ax.quiver(X, Y, U, V, alpha=0.6)
        ax.set_title(f'Vector Field at t={t_val}')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to {save_path}")


def main():
    """主函数: 演示 Flow Matching 的训练和采样"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Creating dataset...")
    data = create_2d_dataset(n_samples=20000, dataset_type='swiss_roll')

    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=512, shuffle=True)

    print("Initializing model...")
    model = UNet1D(input_dim=2, hidden_dim=256, time_emb_dim=128)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    flow_matching = FlowMatching(model, device=device)

    print("\nTraining...")
    losses = flow_matching.train(data_loader, num_epochs=200, lr=1e-3)

    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    print("Training loss saved to training_loss.png")

    print("\nGenerating samples and visualizing...")
    visualize_results(flow_matching, data.numpy())

    print("\nDone!")


if __name__ == '__main__':
    main()
