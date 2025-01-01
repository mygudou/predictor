import torch
import torch.nn as nn


class HybridTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=8, num_layers=4, lstm_hidden=128, dropout=0.2):
        super(HybridTimeSeriesModel, self).__init__()
        # Transformer部分
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, d_model))
        self.transformer = nn.Transformer(d_model=d_model, nhead=n_heads, num_encoder_layers=num_layers,
                                          dropout=dropout)
        # LSTM部分
        self.lstm = nn.LSTM(d_model, lstm_hidden, batch_first=True, num_layers=2, dropout=dropout)
        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden // 2, 1)
        )

    def forward(self, x):
        # Transformer部分
        x = self.embedding(x)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x, x)

        # LSTM部分
        x, _ = self.lstm(x)

        # 输出部分
        x = self.fc_out(x[:, -1, :])  # 取LSTM最后一个时间步
        return x
