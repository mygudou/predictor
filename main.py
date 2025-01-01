from src.database import MongoDBHandler
from src.preprocessing import load_and_preprocess_data
from src.n_beats import NBEATSModel
from src.train import train_model
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 初始化 MongoDB 处理器
    db_handler = MongoDBHandler()

    # 数据窗口设置
    window_size = 60
    forecast_horizon = 120

    # 加载并预处理数据
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess_data(db_handler, window_size)

    # 定义模型
    model = NBEATSModel(input_dim=window_size, forecast_horizon=forecast_horizon)

    # 检查设备 (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 将数据移动到设备上
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # 模型训练
    epochs = 50
    batch_size = 64
    learning_rate = 0.0005
    train_model(model, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, device=device)

    # 模型评估
    model.eval()
    predictions = []
    with torch.no_grad():
        current_input = X_test[0].unsqueeze(0)  # (1, window_size)
        for _ in range(forecast_horizon):  # Predict for forecast_horizon steps
            pred = model(current_input)
            predictions.append(pred[:, -1, :].cpu().numpy())  # only use the last prediction part

            # 更新滑动窗口
            pred_tensor = torch.tensor(pred[:, -1, :], dtype=torch.float32).unsqueeze(0).to(device)
            current_input = torch.cat((current_input[:, 1:, :], pred_tensor), dim=1)

    # 恢复预测值和真实值
    predictions = np.array(predictions).reshape(-1, 1)
    predictions_rescaled = scaler.inverse_transform(predictions)
    real_values_rescaled = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))

    # 打印对比结果
    print("\nComparison of Real vs Predicted for Test Data:")
    for i, (real, pred) in enumerate(zip(real_values_rescaled, predictions_rescaled)):
        print(f"Day {i + 1}: Real: {real[0]:.2f}, Predicted: {pred[0]:.2f}")

    # 可视化对比
    plt.figure(figsize=(15, 6))
    plt.plot(real_values_rescaled, label='Real Data', color='blue')
    plt.plot(predictions_rescaled, label='Predicted Data', color='orange')
    plt.title("Real vs Predicted Stock Prices for Test Data")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

    # 预测未来120天的走势
    future_predictions = []
    with torch.no_grad():
        for _ in range(forecast_horizon):  # 预测未来120天
            pred = model(current_input)
            future_predictions.append(pred[:, -1, :].cpu().numpy())  # only use the last prediction part

            # 更新滑动窗口
            pred_tensor = torch.tensor(pred[:, -1, :], dtype=torch.float32).unsqueeze(0).to(device)
            current_input = torch.cat((current_input[:, 1:, :], pred_tensor), dim=1)

    # 恢复未来预测值
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions_rescaled = scaler.inverse_transform(future_predictions)

    # 打印未来120天预测
    print("\nFuture Predictions for Next 120 Days:")
    for i, pred in enumerate(future_predictions_rescaled):
        print(f"Day {i + 1}: Predicted: {pred[0]:.2f}")

    # 可视化未来预测
    plt.figure(figsize=(15, 6))
    plt.plot(range(len(real_values_rescaled)), real_values_rescaled, label='Real Data', color='blue')
    plt.plot(range(len(real_values_rescaled), len(real_values_rescaled) + len(future_predictions_rescaled)),
             future_predictions_rescaled, label='Future Predictions', color='green')
    plt.title("Predicted Stock Prices for Next 120 Days")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

    # 保存模型和预测结果
    torch.save(model.state_dict(), "n_beats_model.pth")
    np.save("future_predictions.npy", future_predictions_rescaled)

if __name__ == "__main__":
    main()
