class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=8, num_layers=4, dropout=0.2):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, d_model))
        self.transformer = nn.Transformer(d_model=d_model, nhead=n_heads, num_encoder_layers=num_layers,
                                          dropout=dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x, x)
        x = self.fc_out(x[:, -1, :])
        return x
