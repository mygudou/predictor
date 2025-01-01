import numpy as np
import torch


def predict_future(model, scaler, initial_input, future_steps, device, noise_factor=0.01):
    predictions = []

    # 确保 initial_input 在正确设备上
    current_input = initial_input.clone().to(device)

    with torch.no_grad():
        for _ in range(future_steps):
            # 模型预测
            pred = model(current_input)  # pred.shape should be (1, 1, forecast_horizon)
            pred_scalar = pred[:, -1, 0]  # Extract the last forecast step
            predictions.append(pred_scalar.cpu().numpy())

            # 添加噪声并更新输入
            pred_array = torch.tensor([[pred_scalar + np.random.normal(0, noise_factor)]], dtype=torch.float32).to(device)
            current_input = torch.cat((current_input[:, 1:, :], pred_array.unsqueeze(1)), dim=1)

    # 将预测值转换为 NumPy 数组，逆缩放
    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)
