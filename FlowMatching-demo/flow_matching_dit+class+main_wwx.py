import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class FlowMatching:
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
        batch_size, _ = x1.shape
        t = torch.rand(batch_size, device=self.device)
        t = t.view(-1, 1)
        xt = x0 * (1 - t) + x1 * t
        gt_flow = x1 - x0
        pred_flow = self.model(xt, t)
        loss = F.mse_loss(pred_flow, gt_flow)
        return loss

    def train(self, data_loader, num_epochs=200, lr=1e-3):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        losses = []

        for epoch in range(num_epochs):
            for batch_data in data_loader:
                x1 = batch_data[0].to(self.device)
                x0 = torch.randn_like(x1)
                loss = self.compute_loss(x0, x1)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())  # https://www.qianwen.com/share/chat/6c3719e2bc2a4fd0b74aa8ea84b825d8

        return losses

    @torch.no_grad()
    def sample(self, batch_size, num_steps=100):
        """
        使用欧拉方法生成样本
        batch_size: 生成的样本数量
        num_steps: ODE 求解的步数
        """
        self.model.eval()

        x = torch.randn([batch_size, self.model.input_dim], device=self.device)
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.ones(batch_size, device=self.device) * (step * dt)
            pred_flow = self.model(x, t)
            x = x + pred_flow * dt

        self.model.train()
        return x

class SimpleDiT(nn.Module):
    """简化的 DiT 模型"""
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        """
        x: [batch_size, input_dim]
        t: [batch_size] - 时间步，范围 [0, 1]
        """
        t = t.view(-1, 1)
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)


def create_swiss_roll(n_samples=5000):
    """创建 Swiss Roll 数据集"""
    t = np.random.rand(n_samples) * 4 * np.pi
    x = t * np.cos(t)
    y = t * np.sin(t)
    data = np.stack([x, y], axis=1)
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    data = data * 2.0
    return torch.tensor(data, dtype=torch.float32)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建数据集
    data = create_swiss_roll(n_samples=5000) # data = [n_samples, 2]
    dataset = TensorDataset(data)
    data_loader = DataLoader(
        dataset, 
        batch_size=256, 
        shuffle=True)

    # 创建模型
    model = SimpleDiT(input_dim=2, hidden_dim=128)

    # 创建 Flow Matching 实例
    flow_matching = FlowMatching(model, device=device)

    # 训练
    losses = flow_matching.train(data_loader, num_epochs=50, lr=1e-3)

    # 生成样本
    generated = flow_matching.sample(1000, num_steps=50).cpu().numpy()

    print("\nDone!")

