import torch.nn as nn

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