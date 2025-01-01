import numpy as np
import torch

def predict_future(model, scaler, initial_input, future_steps, device="cuda"):
    """
    使用训练好的模型预测未来的值。

    :param model: 已训练的 PyTorch 模型
    :param scaler: 用于逆归一化的 MinMaxScaler
    :param initial_input: 最初的输入序列 (shape: [1, window_size, feature_dim])
    :param future_steps: 要预测的未来步数
    :param device: 使用的设备 ("cuda" 或 "cpu")
    :return: 预测的未来值 (shape: [future_steps, 1])
    """
    predictions = []
    current_input = torch.tensor(initial_input, dtype=torch.float32).to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(future_steps):
            # 模型预测
            pred = model(current_input).item()
            predictions.append(pred)

            # 将预测值扩展为三维，添加到当前输入序列中
            pred_array = torch.tensor([[[pred]]], dtype=torch.float32).to(device)
            current_input = torch.cat((current_input[:, 1:, :], pred_array), dim=1)

    # 逆归一化返回
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
