import numpy as np
import torch

def predict_future(model, scalers, initial_input, future_steps, device='cpu'):
    model.eval()
    predictions = []
    current_input = initial_input.copy()

    for _ in range(future_steps):
        with torch.no_grad():
            current_tensor = torch.tensor(current_input, dtype=torch.float32).to(device)
            pred = model(current_tensor).item()
        predictions.append(pred)

        # 更新输入，使用上一时间步的预测结果替换 Close 特征
        new_input = current_input[:, -1, :].copy()  # [1, feature_dim]
        new_input[0, 0] = pred  # 只替换 Close 的预测值

        # 动态更新相关特征（如 MA_5, MA_8, MA_21, MA_55, MA_144, MA_233）
        for ma_idx, ma_window in zip([5, 6, 7, 8, 9, 10], [5, 8, 21, 55, 144, 233]):
            if current_input.shape[1] >= ma_window:
                new_ma = np.mean(current_input[0, -ma_window:, 0])
            else:
                new_ma = np.mean(current_input[0, :, 0])
            new_input[0, ma_idx] = new_ma  # 更新 MA 特征

        pred_array = new_input.reshape(1, 1, -1)  # [1, 1, feature_dim]

        # 滑动窗口更新输入数据
        current_input = np.append(current_input[:, 1:, :], pred_array, axis=1)

    # 逆标准化
    predictions = np.array(predictions).reshape(-1, 1)
    extended_predictions = np.zeros((future_steps, initial_input.shape[2]))
    extended_predictions[:, 0] = predictions[:, 0]  # 填充 Close 特征

    # 使用最后一时间步的非 Close 特征填充其他特征
    for i in range(1, extended_predictions.shape[1]):
        extended_predictions[:, i] = initial_input[0, -1, i]

    # 逆标准化每个特征
    for i, scaler_key in enumerate(
            ['Close', 'High', 'Low', 'Open', 'Volume', 'MA_5', 'MA_8', 'MA_21', 'MA_55', 'MA_144', 'MA_233', 'Price_Change_Rate', 'Volatility',
             'Volume_Change_Rate', 'Day_sin', 'Day_cos']):
        scaler = scalers[scaler_key]  # 从 scalers 字典获取相应的 scaler

        extended_predictions[:, i] = scaler.inverse_transform(
            extended_predictions[:, i].reshape(-1, 1)
        ).flatten()

    # 返回逆标准化后的 Close 特征（即预测的收盘价）
    return extended_predictions[:, 0]
