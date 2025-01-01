import torch
import torch.nn as nn


class NBEATSModel(nn.Module):
    def __init__(self, input_dim, forecast_horizon=120, stack_layers=3, layer_width=256):
        super(NBEATSModel, self).__init__()
        self.input_dim = input_dim
        self.forecast_horizon = forecast_horizon
        self.stack_layers = stack_layers

        # 创建多个 Block
        self.blocks = nn.ModuleList([
            NBEATSBlock(input_dim, layer_width, forecast_horizon)
            for _ in range(stack_layers)
        ])

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)  # 调整形状为 (batch_size, input_dim)
        elif x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, but got {x.shape[1]}")

        residual = x
        forecast = 0
        for block in self.blocks:
            block_output, residual = block(residual)
            forecast += block_output
        return forecast


class NBEATSBlock(nn.Module):
    def __init__(self, input_dim, layer_width, forecast_horizon):
        super(NBEATSBlock, self).__init__()
        self.input_dim = input_dim
        self.forecast_horizon = forecast_horizon

        self.fc_stack = nn.Sequential(
            nn.Linear(input_dim, layer_width),
            nn.ReLU(),
            nn.Linear(layer_width, layer_width),
            nn.ReLU(),
            nn.Linear(layer_width, input_dim + forecast_horizon)
        )

    def forward(self, x):
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, but got {x.shape[1]}")

        # 前向传播
        fc_output = self.fc_stack(x)
        backcast_output = fc_output[:, :self.input_dim]
        forecast_output = fc_output[:, self.input_dim:]

        # 更新残差
        residual = x - backcast_output
        return forecast_output, residual
