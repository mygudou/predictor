import numpy as np
import torch

def predict_future(model, scaler, initial_input, future_steps, device='cpu'):
    model.eval()  # Set model to evaluation mode
    predictions = []
    current_input = initial_input.copy()  # Shape: [1, seq_len, feature_dim]

    for _ in range(future_steps):
        with torch.no_grad():
            # 模型预测
            current_tensor = torch.tensor(current_input, dtype=torch.float32).to(device)
            pred = model(current_tensor).item()  # 获取标量预测值
        predictions.append(pred)

        # 构造与 current_input 匹配的预测特征向量
        pred_array = np.zeros((1, 1, current_input.shape[2]))  # Shape: [1, 1, feature_dim]
        pred_array[0, 0, 0] = pred  # 将预测值放置在特定的特征维度

        # 更新输入，移除最旧时间步，添加新预测
        current_input = np.append(current_input[:, 1:, :], pred_array, axis=1)

    # 逆归一化之前，将预测结果扩展为具有相同特征维度的形状
    predictions = np.array(predictions).reshape(-1, 1)  # Shape: [future_steps, 1]
    extended_predictions = np.zeros((future_steps, initial_input.shape[2]))  # Shape: [future_steps, feature_dim]
    extended_predictions[:, 0] = predictions[:, 0]  # 将预测值填入收盘价特征位置

    # 逆归一化返回预测值
    return scaler.inverse_transform(extended_predictions)[:, 0]  # 仅返回收盘价
