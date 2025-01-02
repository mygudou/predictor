import torch
import torch.nn as nn
import math

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = self.generate_positional_encoding(1000, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=n_heads, num_encoder_layers=num_layers, dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, 1)

    def generate_positional_encoding(self, seq_len, d_model):
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)
        x = self.transformer(x, x)
        x = self.fc_out(x[:, -1, :])
        return x