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
    X, y, scaler = load_and_preprocess_data(db_handler, window_size)

    # 定义模型
    model = TimeSeriesTransformer(input_dim=1, d_model=256, n_heads=8, num_layers=4)

    # 检查设备 (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 将模型移动到设备

    # 模型训练
    epochs = 50
    batch_size = 64
    learning_rate = 0.0005
    train_model(model, X, y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, device=device)

    # 预测未来数据
    future_steps = 20
    initial_input = torch.tensor(X[-1], dtype=torch.float32).reshape(1, window_size, 1).to(device)
    predictions = predict_future(model, scaler, initial_input, future_steps, device=device)

    # 输出预测结果
    print("Future Predictions:", predictions)

if __name__ == "__main__":
    main()
