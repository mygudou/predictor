import torch.nn as nn
import torch


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers, dropout=0.1, max_seq_len=1000):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)  # 输入特征到 d_model 的线性变换
        self.positional_encoding = nn.Embedding(max_seq_len, d_model)  # 可学习位置编码
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=n_heads, num_encoder_layers=num_layers, dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, 1)  # 输出收盘价

    def forward(self, x):
        # x: [batch_size, sequence_length, feature_dim]
        x = self.embedding(x)  # [batch_size, sequence_length, d_model]
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, sequence_length]
        x = x + self.positional_encoding(positions)  # 添加位置编码
        x = self.transformer(x.permute(1, 0, 2), x.permute(1, 0, 2))  # [seq_len, batch_size, d_model]
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, d_model]

        # 获取最后一个时间步的输出
        x = self.fc_out(x[:, -1, :])  # 仅保留最后一个时间步的输出

        # 返回一个 1D 张量，包含预测的收盘价
        return x
