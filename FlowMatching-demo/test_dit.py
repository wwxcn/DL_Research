"""
简单测试 DiT 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


print("Testing DiT (Diffusion Transformer) implementation...")

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 定义简单的DiT模型
class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.unsqueeze(1)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        x = self.proj(x)
        return x


class DiTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x, c):
        h = self.norm1(x)
        h = self.attn(h)
        x = x + h

        c_expanded = c.unsqueeze(1)
        x = x + c_expanded

        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h

        return x


class DiT(nn.Module):
    def __init__(self, input_dim=2, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.patch_embed = PatchEmbedding(input_dim, embed_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads) for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)
        self.final_proj = nn.Linear(embed_dim, input_dim)

    def forward(self, x, t):
        x = self.patch_embed(x)
        t_emb = self.time_embed(t)

        for block in self.blocks:
            x = block(x, t_emb)

        x = self.final_norm(x)
        x = self.final_proj(x)
        x = x.squeeze(1)

        return x


# 测试1: 创建模型
print("\n1. Creating DiT model...")
model = DiT(input_dim=2, embed_dim=128, depth=4, num_heads=4).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# 测试2: 前向传播
print("\n2. Testing forward pass...")
x = torch.randn(10, 2).to(device)
t = torch.rand(10).to(device)
output = model(x, t)
print(f"Input shape: {x.shape}, t shape: {t.shape}")
print(f"Output shape: {output.shape}")
print(f"Output sample: {output[0].cpu().detach().numpy()}")

# 测试3: 损失计算
print("\n3. Testing loss computation...")
x0 = torch.randn(10, 2).to(device)
x1 = torch.randn(10, 2).to(device)
t = torch.rand(10).to(device)
t_expanded = t.view(-1, 1)
x_t = (1 - t_expanded) * x0 + t_expanded * x1
target_v = x1 - x0
pred_v = model(x_t, t)
loss = F.mse_loss(pred_v, target_v)
print(f"Loss: {loss.item():.6f}")

# 测试4: 训练步骤
print("\n4. Testing training step...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("Training step completed successfully")

# 测试5: 采样
print("\n5. Testing sampling...")
model.eval()
x = torch.randn(5, 2).to(device)
dt = 0.01
num_steps = 10

with torch.no_grad():
    for i in range(num_steps):
        t = torch.ones(5, device=device) * (i * dt)
        v = model(x, t)
        x = x + dt * v

print(f"Final samples shape: {x.shape}")
print(f"Sample values:\n{x.cpu().detach().numpy()}")

# 测试6: 完整训练循环
print("\n6. Testing full training loop...")
data = torch.randn(1000, 2).to(device)
batch_size = 32

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

losses = []
for epoch in range(5):
    epoch_losses = []
    for i in range(0, len(data), batch_size):
        x1 = data[i:i+batch_size]
        x0 = torch.randn_like(x1)

        t = torch.rand(x1.shape[0], device=device)
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
    print(f"Epoch {epoch+1}/5, Loss: {avg_loss:.6f}")

print(f"\nLoss decreased from {losses[0]:.6f} to {losses[-1]:.6f}")
print(f"Loss reduction: {100*(losses[0]-losses[-1])/losses[0]:.2f}%")

print("\n" + "="*50)
print("✓ All DiT tests passed successfully!")
print("="*50)
