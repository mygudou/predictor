from src.database import MongoDBHandler
from src.preprocessing import load_and_preprocess_data
from src.model import TimeSeriesTransformer
from src.train import train_model
from src.predict import predict_future
import torch


def main():
    # 数据库处理
    db_handler = MongoDBHandler()

    # 数据加载与预处理
    window_size = 60
    X, y, scaler = load_and_preprocess_data(db_handler, window_size)

    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型构建
    model = TimeSeriesTransformer(input_dim=1, d_model=64, n_heads=4, num_layers=2).to(device)

    # 模型训练
    train_model(model, X, y, epochs=50, device=device)

    # 预测
    future_steps = 12
    initial_input = X[-1].reshape(1, window_size, 1)
    predictions = predict_future(model, scaler, initial_input, future_steps, device=device)

    print(predictions)


if __name__ == "__main__":
    main()
