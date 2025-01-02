import numpy as np
import torch

def predict_future(model, scaler, initial_input, future_steps, device="cpu"):
    predictions = []
    current_input = initial_input.copy()  # 确保不修改原始输入

    for _ in range(future_steps):
        with torch.no_grad():
            # 模型预测
            pred = model(torch.tensor(current_input, dtype=torch.float32).to(device)).item()
        predictions.append(pred)

        # 将预测值扩展为三维，添加到当前输入序列中
        pred_array = np.array([[[pred]]])  # shape: [1, 1, 1]
        current_input = np.append(current_input[:, 1:, :], pred_array, axis=1)

    # 逆归一化返回
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
