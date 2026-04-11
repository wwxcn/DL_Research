"""
简化版 Flow Matching - 直接在终端显示结果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


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
    def sample(self, batch_size, num_steps=100):
        self.model.eval()
        x = torch.randn(batch_size, self.model.input_dim, device=self.device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.ones(batch_size, device=self.device) * (i * dt)
            v = self.model(x, t)
            x = x + dt * v

        self.model.train()
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


def main():
    """主函数: 演示 Flow Matching 的训练和采样"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Creating dataset...")
    data = create_2d_dataset(n_samples=20000, dataset_type='swiss_roll')
    print(f"Dataset created with shape: {data.shape}")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"Data mean: {data.mean():.2f}, std: {data.std():.2f}")

    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    print(f"DataLoader created with {len(data_loader)} batches")

    print("Initializing model...")
    model = UNet1D(input_dim=2, hidden_dim=256, time_emb_dim=128)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    flow_matching = FlowMatching(model, device=device)
    print("FlowMatching instance created")

    print("\n" + "="*50)
    print("Training Flow Matching Model")
    print("="*50 + "\n")

    losses = flow_matching.train(data_loader, num_epochs=200, lr=1e-3)
    print(f"\nTraining completed. Final loss: {losses[-1]:.6f}")
    print(f"Loss decreased from {losses[0]:.6f} to {losses[-1]:.6f} ({100*(losses[0]-losses[-1])/losses[0]:.2f}% reduction)")

    print("\n" + "="*50)
    print("Generating Samples")
    print("="*50 + "\n")

    generated = flow_matching.sample(1000, num_steps=100).cpu().numpy()
    print(f"Generated {len(generated)} samples")
    print(f"Generated data range: [{generated.min():.2f}, {generated.max():.2f}]")
    print(f"Generated data mean: {generated.mean():.2f}, std: {generated.std():.2f}")

    print(f"\nOriginal data statistics:")
    print(f"  Mean: {data.mean().numpy():.2f}, Std: {data.std().numpy():.2f}")
    print(f"Generated data statistics:")
    print(f"  Mean: {generated.mean():.2f}, Std: {generated.std():.2f}")

    print("\n" + "="*50)
    print("Testing with Different Datasets")
    print("="*50 + "\n")

    for dataset_type in ['moons', 'gaussian_mixture']:
        print(f"\nTesting with {dataset_type} dataset...")
        test_data = create_2d_dataset(n_samples=5000, dataset_type=dataset_type)
        test_dataset = TensorDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

        test_model = UNet1D(input_dim=2, hidden_dim=256, time_emb_dim=128)
        test_flow = FlowMatching(test_model, device=device)

        print(f"Training on {dataset_type}...")
        test_losses = test_flow.train(test_loader, num_epochs=100, lr=1e-3)

        test_generated = test_flow.sample(500, num_steps=100).cpu().numpy()
        print(f"Final loss: {test_losses[-1]:.6f}")
        print(f"Loss reduction: {100*(test_losses[0]-test_losses[-1])/test_losses[0]:.2f}%")
        print(f"Generated samples: {len(test_generated)}")

    print("\n" + "="*50)
    print("All Tests Completed Successfully!")
    print("="*50)


if __name__ == '__main__':
    main()
