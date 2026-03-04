import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = q.size(0)
        
        q = self.split_heads(self.W_q(q))
        k = self.split_heads(self.W_k(k))
        v = self.split_heads(self.W_v(v))
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x

'''
softmax 是在 generate 方法里做的 ，而不是在 forward 里。为什么这样设计？
训练流程：
  logits = model(x)           # [batch, seq, vocab]
  loss = CrossEntropyLoss(logits.view(-1, vocab), labels.view(-1))
  # CrossEntropyLoss 内部: log_softmax → negative log likelihood
生成流程：
  logits = model(x)           # [batch, seq, vocab]
  probs = softmax(logits)     # 转成概率
  token = sample(probs)       # 采样

分离的好处：
1. forward 保持简洁，返回原始logits
2. 训练时直接用logits算loss，避免多余的softmax计算
3. 生成时根据需要选择采样策略（greedy、top-k、top-p等）
'''
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = x.size()
        
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.fc(x) # [batch_size, sequence_length, vocab_size]
        return x

    def generate_subsequent_mask(seq_len: int) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = None,
        eos_token_id: int = None
    ) -> torch.Tensor:
        """
        自回归生成：- 从起始token（如 <bos> ）开始，每次预测下一个token，
                  - 将预测结果拼接到输入，继续预测，直到遇到结束token（如 <eos> ）或达到最大长度

        Args:
            input_ids: [batch_size, seq_len] 起始token序列
            max_new_tokens: 最多生成多少个新token
            temperature: 采样温度，控制随机性（>1增加随机性，<1减少随机性）
            top_k: 只从概率最高的k个token中采样
            eos_token_id: 结束token的id，遇到则停止

        Returns:
            [batch_size, seq_len + generated_len] 生成的完整token序列

        关键概念：
            logits: 模型的原始输出，未经过softmax的原始分数
                    表示每个token的“原始可能性”，值越大表示越可能
        重点解释：
            - 训练时，输入: ["我", "爱", "你"]，对应的标签: ["爱", "你", "！"]，即输出结果是下一个token的预测
            - 因此，推理时，每个位置的输出实际上是在预测该位置之后的下一个 token
        """
        self.eval()  # 切换到评估模式
        batch_size = input_ids.size(0)

        for _ in range(max_new_tokens):
            # 1. 获取当前序列长度，生成因果mask（确保只能看到已生成的token）
            seq_len = input_ids.size(1)
            mask = self.generate_subsequent_mask(seq_len).to(input_ids.device)

            # 2. 前向传播获取logits（原始输出分数）
            logits = self.forward(input_ids, mask)  # [batch, seq_len, vocab_size]
            
            # 3. 只取最后一个位置的logits（预测下一个token）
            next_token_logits = logits[:, -1, :]  # [batch, vocab_size]
            
            # 4. 应用温度参数调整分布（控制随机性）
            next_token_logits = next_token_logits / temperature

            # 5. 可选：应用top-k采样（只考虑概率最高的k个token）
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, top_k)  # 获取前k个最大值
                # 将低于第k个值的logits设为负无穷（排除在外）
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('inf')

            # 6. 将logits转换为概率分布
            probs = F.softmax(next_token_logits, dim=-1)  # [batch, vocab_size]
            
            # 7. 按概率采样下一个token
            # torch.multinomial按概率分布进行采样,torch.argmax按最大值采样
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]

            # 8. 将新token拼接到序列末尾
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # 9. 检查是否生成了结束token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids

