import torch.nn as nn
import torch


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)  # Adjust for multi-feature input
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, d_model))  # Positional encoding
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=n_heads, num_encoder_layers=num_layers, dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, 1)  # Single output

    def forward(self, x):
        # x: [batch_size, sequence_length, feature_dim]
        x = self.embedding(x)  # [batch_size, sequence_length, d_model]
        x = x + self.positional_encoding[:, :x.size(1), :]  # Add positional encoding
        x = self.transformer(x.permute(1, 0, 2), x.permute(1, 0, 2))  # [seq_len, batch_size, d_model]
        x = x.permute(1, 0, 2)  # Back to [batch_size, seq_len, d_model]
        x = self.fc_out(x[:, -1, :])  # Take the last time step
        return x
