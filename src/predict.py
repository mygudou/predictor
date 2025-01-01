import numpy as np
import torch


def predict_future(model, scaler, initial_input, future_steps, noise_factor=0.01):
    predictions = []

    # 确保 `initial_input` 是 PyTorch 张量
    current_input = initial_input.clone()  # 使用 PyTorch 的 `clone()` 方法代替 `copy()`

    for _ in range(future_steps):
        with torch.no_grad():
            # 模型预测，确保输入在正确的设备上
            pred = model(current_input).item()
        predictions.append(pred)

        # 添加噪声并更新输入
        pred_array = torch.tensor([[[pred + np.random.normal(0, noise_factor)]]], dtype=torch.float32).to(
            current_input.device)
        current_input = torch.cat((current_input[:, 1:, :], pred_array), dim=1)

    # 将预测值转换为 NumPy 数组，逆缩放
    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)
