from src.database import MongoDBHandler
from src.preprocessing import load_and_preprocess_data
from src.model import TimeSeriesTransformer
from src.train import train_model
from src.predict import predict_future
import torch

def main():
    # 初始化 MongoDB 处理器
    db_handler = MongoDBHandler()

    # 数据窗口设置
    window_size = 60

    # 加载并预处理数据
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data(db_handler, window_size)

    # 定义模型
    model = TimeSeriesTransformer(input_dim=1, d_model=256, n_heads=8, num_layers=4)

    # 检查设备 (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 模型训练
    epochs = 50
    batch_size = 64
    learning_rate = 0.0005
    train_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, device=device)

    # 预测最近一年的走势
    model.eval()
    predictions = []
    with torch.no_grad():
        current_input = torch.tensor(X_test[0], dtype=torch.float32).reshape(1, window_size, 1).to(device)
        for _ in range(len(y_test)):
            pred = model(current_input).item()
            predictions.append(pred)

            # 更新当前输入
            pred_array = torch.tensor([[[pred]]], dtype=torch.float32).to(device)
            current_input = torch.cat((current_input[:, 1:, :], pred_array), dim=1)

    # 恢复预测值和真实值
    predictions = np.array(predictions).reshape(-1, 1)
    predictions_rescaled = scaler.inverse_transform(predictions)
    real_values_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 打印对比结果
    for i, (real, pred) in enumerate(zip(real_values_rescaled, predictions_rescaled)):
        print(f"Day {i + 1}: Real: {real[0]:.2f}, Predicted: {pred[0]:.2f}")

    # 可视化对比
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 6))
    plt.plot(real_values_rescaled, label='Real Data', color='blue')
    plt.plot(predictions_rescaled, label='Predicted Data', color='orange')
    plt.title("Real vs Predicted Stock Prices for Last Year")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

