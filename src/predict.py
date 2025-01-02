import numpy as np
import torch


def predict_future(model, scaler, initial_input, future_steps):
    predictions = []
    current_input = initial_input.copy()  # 确保不修改原始输入

    for _ in range(future_steps):
        with torch.no_grad():
            # 模型预测
            pred = model(torch.tensor(current_input, dtype=torch.float32)).item()
        predictions.append(pred)

        # 将预测值扩展为三维，添加到当前输入序列中
        pred_array = np.array([[[pred] + [0] * (current_input.shape[-1] - 1)]])  # 用预测值填充第一列，其他特征置为零
        current_input = np.append(current_input[:, 1:, :], pred_array, axis=1)

    # 逆归一化返回
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
