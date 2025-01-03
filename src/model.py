# model.py
import torch.nn as nn
import torch

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers, dropout=0.1, max_seq_len=1000):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Embedding(max_seq_len, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=n_heads, num_encoder_layers=num_layers, dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.positional_encoding(positions)
        x = self.transformer(x.permute(1, 0, 2), x.permute(1, 0, 2))
        x = x.permute(1, 0, 2)
        x = self.fc_out(x[:, -1, :])
        return x


class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
        super(TimeSeriesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x.shape = [batch_size, seq_len, input_dim]
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out.shape = [batch_size, seq_len, hidden_dim]
        # 只取最后一个时间步的输出
        out = self.fc_out(lstm_out[:, -1, :])  # out.shape = [batch_size, 1]
        return out