# main.py
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
    X, y, scalers = load_and_preprocess_data(db_handler, window_size)

    # 划分训练集与验证集
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模型构建
    model = TimeSeriesTransformer(input_dim=17, d_model=128, n_heads=4, num_layers=2).to(device)

    # 模型训练
    train_model(model, X_train, y_train, X_val, y_val, epochs=60, device=device)

    # 预测
    future_steps = 12
    initial_input = X[-1].reshape(1, window_size, 17)
    predictions = predict_future(model, scalers, initial_input, future_steps, device=device)

    print(predictions)

if __name__ == "__main__":
    main()
