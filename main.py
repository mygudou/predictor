from src.database import MongoDBHandler
from src.preprocessing import load_and_preprocess_data
from src.model import TimeSeriesTransformer
from src.train import train_model
from src.predict import predict_future
import torch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据库处理
    db_handler = MongoDBHandler()

    # 数据加载与预处理
    window_size = 60
    X, y, scaler = load_and_preprocess_data(db_handler, window_size)

    # 模型构建
    model = TimeSeriesTransformer(input_dim=1, d_model=64, n_heads=4, num_layers=2)

    # 模型训练
    train_model(model, X, y, epochs=50, device=device)

    # 预测未来12天
    future_steps = 12
    initial_input = X[-1].reshape(1, window_size, 1)
    future_predictions = predict_future(model, scaler, initial_input, future_steps, device=device)

    # 模型对历史数据的测试对比
    model.eval()
    test_inputs = torch.tensor(X[-future_steps:], dtype=torch.float32).to(device)
    with torch.no_grad():
        historical_predictions = model(test_inputs).cpu().numpy()
    historical_predictions = scaler.inverse_transform(historical_predictions)
    historical_actual = scaler.inverse_transform(y[-future_steps:].reshape(-1, 1))

    # 输出结果
    print("\nFuture Predictions:")
    print("Time Step\tPredicted")
    for i, pred in enumerate(future_predictions):
        print(f"{i + 1}\t{pred[0]:.4f}")

    print("\nHistorical Test Comparison:")
    print("Time Step\tActual\tPredicted")
    for i, (act, pred) in enumerate(zip(historical_actual, historical_predictions)):
        print(f"{i + 1}\t{act[0]:.4f}\t{pred[0]:.4f}")


if __name__ == "__main__":
    main()
