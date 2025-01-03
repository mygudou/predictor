from src.database import MongoDBHandler
from src.preprocessing import load_and_preprocess_data
from src.tft_model import TemporalFusionTransformer
from src.train import train_model
from src.predict import predict_future
import torch
import os

# 启用内存扩展
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    # 数据库处理
    db_handler = MongoDBHandler()

    # 数据加载与预处理
    window_size = 60
    X, y, scalers, static_features = load_and_preprocess_data(db_handler, window_size)

    # 划分训练集与验证集
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    static_features_train, static_features_val = static_features[:split_idx], static_features[split_idx:]

    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 将静态特征转为 tensor，并传递到设备
    static_features_train = torch.tensor(static_features_train, dtype=torch.float32).to(device)
    static_features_val = torch.tensor(static_features_val, dtype=torch.float32).to(device)

    # 模型构建
    model = TemporalFusionTransformer(input_dim=16, static_dim=static_features.shape[1], hidden_dim=64, num_heads=4, num_layers=2).to(device)

    # 模型训练
    train_model(model, X_train, y_train, X_val, y_val, static_features_train, static_features_val, epochs=100, device=device)

    # 预测
    future_steps = 128
    initial_input = X[-1].reshape(1, window_size, 16)  # 使用最后一个窗口作为输入
    static_feature = static_features[-1].reshape(1, -1)  # 使用最后一个静态特征
    predictions = predict_future(model, scalers, initial_input, static_feature, future_steps, device=device)

    print(predictions)

if __name__ == "__main__":
    main()
