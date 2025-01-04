# model.py
import torch.nn as nn
import torch

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers, dropout=0.3, max_seq_len=1000):
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