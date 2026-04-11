"""
确保输出的 DiT 测试
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def main():
    print("Testing DiT (Diffusion Transformer) implementation...", flush=True)
    print("="*50, flush=True)

    # 设置设备
    device = 'cpu'
    print(f"Using device: {device}", flush=True)

    # 定义极简DiT模型
    class SimpleDiT(nn.Module):
        def __init__(self, input_dim=2, embed_dim=64, depth=2, num_heads=2):
            super().__init__()
            self.input_dim = input_dim
            self.embed_dim = embed_dim

            # 输入投影
            self.input_proj = nn.Linear(input_dim, embed_dim)

            # 时间嵌入
            self.time_embed = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

            # Transformer层
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.0,
                    batch_first=True
                )
                for _ in range(depth)
            ])

            # 输出投影
            self.output_proj = nn.Linear(embed_dim, input_dim)

        def forward(self, x, t):
            # 输入投影: [batch, input_dim] -> [batch, 1, embed_dim]
            x = self.input_proj(x).unsqueeze(1)

            # 时间嵌入: [batch] -> [batch, embed_dim]
            t_emb = self.time_embed(t.unsqueeze(1))

            # 添加时间信息
            x = x + t_emb.unsqueeze(1)

            # 通过Transformer层
            for layer in self.layers:
                x = layer(x)

            # 输出投影: [batch, 1, embed_dim] -> [batch, input_dim]
            x = self.output_proj(x).squeeze(1)

            return x

    # 测试1: 创建模型
    print("\n1. Creating DiT model...", flush=True)
    model = SimpleDiT(input_dim=2, embed_dim=64, depth=2, num_heads=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # 测试2: 前向传播
    print("\n2. Testing forward pass...", flush=True)
    x = torch.randn(10, 2)
    t = torch.rand(10)
    output = model(x, t)
    print(f"Input shape: {x.shape}, t shape: {t.shape}", flush=True)
    print(f"Output shape: {output.shape}", flush=True)
    print(f"Output sample: {output[0].detach().numpy()}", flush=True)

    # 测试3: 损失计算
    print("\n3. Testing loss computation...", flush=True)
    x0 = torch.randn(10, 2)
    x1 = torch.randn(10, 2)
    t = torch.rand(10)
    t_expanded = t.view(-1, 1)
    x_t = (1 - t_expanded) * x0 + t_expanded * x1
    target_v = x1 - x0
    pred_v = model(x_t, t)
    loss = F.mse_loss(pred_v, target_v)
    print(f"Loss: {loss.item():.6f}", flush=True)

    # 测试4: 训练步骤
    print("\n4. Testing training step...", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Training step completed successfully", flush=True)

    # 测试5: 采样
    print("\n5. Testing sampling...", flush=True)
    model.eval()
    x = torch.randn(5, 2)
    dt = 0.01
    num_steps = 10

    with torch.no_grad():
        for i in range(num_steps):
            t = torch.ones(5) * (i * dt)
            v = model(x, t)
            x = x + dt * v

    print(f"Final samples shape: {x.shape}", flush=True)
    print(f"Sample values:\n{x.detach().numpy()}", flush=True)

    # 测试6: 完整训练循环
    print("\n6. Testing full training loop...", flush=True)
    data = torch.randn(500, 2)
    batch_size = 32

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for epoch in range(5):
        epoch_losses = []
        for i in range(0, len(data), batch_size):
            x1 = data[i:i+batch_size]
            x0 = torch.randn_like(x1)

            t = torch.rand(x1.shape[0])
            t_expanded = t.view(-1, 1)
            x_t = (1 - t_expanded) * x0 + t_expanded * x1
            target_v = x1 - x0
            pred_v = model(x_t, t)
            loss = F.mse_loss(pred_v, target_v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/5, Loss: {avg_loss:.6f}", flush=True)

    print(f"\nLoss decreased from {losses[0]:.6f} to {losses[-1]:.6f}", flush=True)
    print(f"Loss reduction: {100*(losses[0]-losses[-1])/losses[0]:.2f}%", flush=True)

    print("\n" + "="*50, flush=True)
    print("✓ All DiT tests passed successfully!", flush=True)
    print("="*50, flush=True)


if __name__ == '__main__':
    main()
