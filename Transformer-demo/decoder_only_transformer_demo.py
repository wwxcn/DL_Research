import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int) -> None:
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.mask = torch.tril(torch.ones((1, 1, self.d_model, self.d_model)))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # input : (batch_size, seq_len, d_model)
        # output: (batch_size, h, seq_len, d_k)
        b, n, _ = x.size()
        return x.view(b, n, self.h, self.d_k).transpose(1, 2) # 最开始没写出来，不需要contiguous()？
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # input : (batch_size, h, seq_len, d_k)
        # output: (batch_size, seq_len, d_model)
        b, _, n, _ = x.size()
        return x.transpose(1, 2).view(b, n, self.d_model) # 不需要contiguous()，因为view()不会改变内存布局
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        b, n, _ = x.size()
        q = self.split_heads(self.W_q(x))
        k = self.split_heads(self.W_k(x))
        v = self.split_heads(self.W_v(x))
        score = q @ k.transpose(-2, -1)
        if self.mask is not None:
            score = score.masked_fill(self.mask == 0, float('-inf'))
        score = self.softmax((score)/torch.sqrt(self.d_k))
        score = self.dropout(score)
        out = score @ v
        out = self.combine_heads(out)
        out = self.W_o(out)
        out = self.dropout(out)
        return out
    
class FeedForwardNet(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU() # 可以采用F.relu()，效果相同

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, hidden_dim, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, h)
        self.norm1 = nn.LayerNorm(d_model) # 需要加上d_model
        self.ff = FeedForwardNet(d_model, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout) # 忘了加dropout
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        origin_x = x
        x = self.mha(x)
        x = self.dropout1(self.norm1(x + origin_x))
        origin_x = x
        x = self.dropout2(self.ff(x))
        x = self.norm2(x + origin_x)
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, d_model, class_num, layer_num, dropout, num_heads, hidden_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(class_num, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.proj = nn.Linear(d_model, class_num)
        self.softmax = nn.Softmax(dim=-1)
        self.seq = nn.Sequential([DecoderLayer(d_model, num_heads, hidden_dim, dropout) for _ in range(layer_num)]) # 忘了加[]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(b, seq_len) # 自己没写出来
        x = self.token_embedding(x) + self.pos_embedding(positions)
        x = self.dropout(x)
        x = self.seq(x) # 待确定是否这样操作
        x = self.proj(x) # (b, n, class_num logits???)
        # x = self.softmax(x) # 豆包：softmax 放在 generate 里做，而不是 forward 里。
        return x