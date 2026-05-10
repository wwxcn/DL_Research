"""
Flow Matching (Rectified Flow) 实现
基于论文: "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


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
