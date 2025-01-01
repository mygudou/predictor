import torch
import torch.nn as nn


class NBEATSModel(nn.Module):
    def __init__(self, input_dim, forecast_horizon=120, stack_layers=3, layer_width=256):
        super(NBEATSModel, self).__init__()
        self.input_dim = input_dim
        self.forecast_horizon = forecast_horizon
        self.stack_layers = stack_layers

        # 基础Block
        self.blocks = nn.ModuleList([
            NBEATSBlock(input_dim, layer_width, forecast_horizon)
            for _ in range(stack_layers)
        ])

    def forward(self, x):
        residual = x
        forecast = 0
        for block in self.blocks:
            block_output, residual = block(residual)
            forecast += block_output
        return forecast


class NBEATSBlock(nn.Module):
    def __init__(self, input_dim, layer_width, forecast_horizon):
        super(NBEATSBlock, self).__init__()
        self.fc_stack = nn.Sequential(
            nn.Linear(input_dim, layer_width),
            nn.ReLU(),
            nn.Linear(layer_width, layer_width),
            nn.ReLU(),
            nn.Linear(layer_width, input_dim + forecast_horizon)
        )
        self.input_dim = input_dim
        self.forecast_horizon = forecast_horizon

    def forward(self, x):
        # 输出预测值和残差
        fc_output = self.fc_stack(x)
        backcast, forecast = fc_output[:, :self.input_dim], fc_output[:, self.input_dim:]
        residual = x - backcast
        return forecast, residual
