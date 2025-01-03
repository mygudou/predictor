import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers, dropout=0.1, max_seq_len=1000):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Embedding(max_seq_len, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,  # 只使用编码器，不需要解码器
            dropout=dropout,
        )

        self.fc_out = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)  # Dropout 防止过拟合

    def forward(self, x):
        # 输入 x: [batch_size, seq_len, input_dim]
        x = self.embedding(x)  # [batch_size, seq_len, d_model]

        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        x = x + self.positional_encoding(positions)  # [batch_size, seq_len, d_model]

        # Transformer 的输入需要 [seq_len, batch_size, d_model]
        x = x.permute(1, 0, 2)  # 转置为 [seq_len, batch_size, d_model]

        # 只使用编码器部分，传入 src，tgt 可以设为 None
        x = self.transformer(src=x, tgt=x)  # 这里传入 src 和 tgt 都是 x，表示使用自注意力机制

        # 输出转换回 [batch_size, seq_len, d_model]
        x = x.permute(1, 0, 2)  # 转置回 [batch_size, seq_len, d_model]

        # 使用最后一个时间步的输出
        x = self.dropout(x[:, -1, :])  # Dropout 防止过拟合
        x = self.fc_out(x)  # [batch_size, 1]
        return x
