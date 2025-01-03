import numpy as np
import torch


def predict_future(model, scaler, initial_input, future_steps, device='cpu'):
    model.eval()  # 设置模型为评估模式
    predictions = []
    current_input = initial_input.copy()  # [1, seq_len, feature_dim]

    for _ in range(future_steps):
        with torch.no_grad():
            current_tensor = torch.tensor(current_input, dtype=torch.float32).to(device)
            pred = model(current_tensor).item()  # 获取预测值（标量）
        predictions.append(pred)

        # 创建新输入特征向量
        pred_array = np.zeros((1, 1, current_input.shape[2]))  # [1, 1, feature_dim]
        pred_array[0, 0, 0] = pred  # 填入预测值

        # 滑动窗口更新输入
        current_input = np.append(current_input[:, 1:, :], pred_array, axis=1)

    # 逆归一化之前扩展预测结果为特征维度
    predictions = np.array(predictions).reshape(-1, 1)  # [future_steps, 1]
    extended_predictions = np.zeros((future_steps, initial_input.shape[2]))  # [future_steps, feature_dim]
    extended_predictions[:, 0] = predictions[:, 0]  # 仅填充收盘价特征

    # 逆归一化并返回收盘价预测值
    return scaler.inverse_transform(extended_predictions)[:, 0]  # 返回收盘价
