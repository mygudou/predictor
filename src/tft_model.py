import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_dim, static_dim, hidden_dim, num_heads, num_layers, dropout=0.1, max_seq_len=1000):
        super(TemporalFusionTransformer, self).__init__()

        # 参数
        self.input_dim = input_dim  # 输入维度
        self.static_dim = static_dim  # 静态特征维度
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.num_heads = num_heads  # Attention Heads数量
        self.num_layers = num_layers  # Transformer层数
        self.max_seq_len = max_seq_len  # 最大序列长度
        self.dropout = dropout  # Dropout比例

        # 1. Embedding层
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Embedding(max_seq_len, hidden_dim)

        # 2. Static Feature Encoding
        self.static_embedding = nn.Linear(static_dim, hidden_dim)

        # 3. Temporal Fusion Layer
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, dropout=dropout)

        # 4. Attention Mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)

        # 5. Transformer层
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers, dropout=dropout
        )

        # 6. Final Fully Connected Layer
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, static_features):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()

        # 1. Embedding and positional encoding
        x = self.embedding(x)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x += self.positional_encoding(pos)

        # 2. Static features embedding
        static_embedding = self.static_embedding(static_features).unsqueeze(1).repeat(1, seq_len, 1)

        # 3. Concatenate the input and static features
        x = x + static_embedding

        # 4. LSTM encoding
        x, _ = self.lstm(x)

        # 5. Attention mechanism
        attn_output, _ = self.attention(x, x, x)

        # 6. Transformer Layer
        x = self.transformer(x.permute(1, 0, 2), x.permute(1, 0, 2))  # Transpose for Transformer
        x = x.permute(1, 0, 2)  # Back to original shape (batch_size, seq_len, hidden_dim)

        # 7. Output layer (prediction)
        out = self.fc_out(x[:, -1, :])  # Only use the last timestep's output

        return out
