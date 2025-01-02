from src.database import MongoDBHandler
from src.preprocessing import load_and_preprocess_data
from src.model import TimeSeriesTransformer
from src.train import train_model
from src.predict import predict_future


def main():
    # 数据库处理
    db_handler = MongoDBHandler()

    # 数据加载与预处理
    window_size = 320
    X, y, scaler = load_and_preprocess_data(db_handler, window_size)

    # 模型构建
    input_dim = X.shape[2]  # 根据特征数量动态设置输入维度
    model = TimeSeriesTransformer(input_dim=input_dim, d_model=64, n_heads=4, num_layers=2)

    # 模型训练
    train_model(model, X, y, epochs=50)

    # 预测
    future_steps = 12
    initial_input = X[-1].reshape(1, window_size, input_dim)
    predictions = predict_future(model, scaler, initial_input, future_steps)

    print(predictions)


if __name__ == "__main__":
    main()
