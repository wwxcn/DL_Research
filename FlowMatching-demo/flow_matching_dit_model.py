"""
Diffusion Transformer (DiT) 模型实现
基于论文: "Scalable Diffusion Models with Transformers"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """
    将输入数据分割成patches并嵌入
    对于2D数据，我们可以将每个维度作为一个patch
    """
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        """
        x: [batch_size, input_dim]
        返回: [batch_size, 1, embed_dim] - 将整个输入作为一个patch
        """
        x = self.proj(x)  # [batch_size, embed_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, embed_dim]
        return x


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [batch_size, seq_len, embed_dim]
        返回: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    """
    前馈神经网络
    """
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DiTBlock(nn.Module):
    """
    DiT Transformer Block
    包含自注意力、条件注意力（时间步）和MLP
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x, c):
        """
        x: [batch_size, seq_len, embed_dim] - 输入特征
        c: [batch_size, embed_dim] - 条件（时间步嵌入）
        返回: [batch_size, seq_len, embed_dim]
        """
        # 自注意力
        h = self.norm1(x)
        h = self.attn(h)
        x = x + h

        # 添加条件信息
        c_expanded = c.unsqueeze(1)  # [batch_size, 1, embed_dim]
        x = x + c_expanded

        # MLP
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h

        return x


class TimeEmbedding(nn.Module):
    """
    时间步嵌入（正弦位置编码）
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [batch_size] - 时间步，范围 [0, 1]
        返回: [batch_size, dim] - 时间嵌入
        """
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) 用于Flow Matching
    """
    def __init__(self, input_dim=2, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Patch嵌入
        self.patch_embed = PatchEmbedding(input_dim, embed_dim)

        # 时间嵌入
        self.time_embed = nn.Sequential(
            TimeEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # 最终层
        self.final_norm = nn.LayerNorm(embed_dim)
        self.final_proj = nn.Linear(embed_dim, input_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x, t):
        """
        前向传播
        x: [batch_size, input_dim] - 当前状态
        t: [batch_size] - 时间步
        返回: [batch_size, input_dim] - 估计的向量场 v(x, t)
        """
        # Patch嵌入
        x = self.patch_embed(x)  # [batch_size, 1, embed_dim]

        # 时间嵌入
        t_emb = self.time_embed(t)  # [batch_size, embed_dim]

        # 通过Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # 最终投影
        x = self.final_norm(x)
        x = self.final_proj(x)  # [batch_size, 1, input_dim]
        x = x.squeeze(1)  # [batch_size, input_dim]

        return x


if __name__ == '__main__':
    # 测试DiT模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing DiT on {device}")
    
    # 创建模型
    model = DiT(input_dim=2, embed_dim=256, depth=6, num_heads=8).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    x = torch.randn(10, 2).to(device)
    t = torch.rand(10).to(device)
    output = model(x, t)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    print("DiT model test passed!")
