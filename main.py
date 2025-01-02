# main.py
from src.database import MongoDBHandler
from src.preprocessing import load_and_preprocess_data
from src.model import TimeSeriesTransformer
from src.train import train_model
from src.predict import predict_future
from sklearn.model_selection import TimeSeriesSplit
import torch

def main():
    # 初始化数据库处理对象，用于从 MongoDB 中加载数据
    db_handler = MongoDBHandler()

    # 定义时间窗口大小，用于生成时间序列数据
    window_size = 60

    # 加载并预处理数据
    # 包括时间序列划分、归一化处理，以及生成用于训练的输入和输出序列
    X, y, scaler = load_and_preprocess_data(db_handler, window_size)

    # 创建时间序列交叉验证器，将数据划分为训练集和验证集
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        # 根据索引划分训练集和测试集
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 初始化时间序列 Transformer 模型
        model = TimeSeriesTransformer(input_dim=1, d_model=64, n_heads=4, num_layers=2)

        # 训练模型，使用早停机制避免过拟合
        train_model(model, X_train, y_train, epochs=50, patience=10)

        # 加载保存的最佳模型权重
        model.load_state_dict(torch.load("best_model.pth"))

        # 准备预测的初始输入数据
        # 使用测试集的第一个样本作为预测起点
        initial_input = X_test[0].reshape(1, window_size, 1)

        # 预测未来数据点，长度与测试集一致
        predictions = predict_future(model, scaler, initial_input, len(y_test))

        # 输出预测结果，通常用于评估模型性能
        print(predictions)

if __name__ == "__main__":
    # 主函数入口，调用主程序
    main()
