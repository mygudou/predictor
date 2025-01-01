import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def load_and_preprocess_data(db_handler, window_size, val_split=0.2):
    """
    加载并预处理数据，包括训练集、验证集和测试集的划分。

    参数:
        db_handler: 数据库处理对象，用于获取数据。
        window_size: 滑动窗口大小。
        val_split: 验证集占训练集比例 (默认0.2)。

    返回:
        X_train, y_train: 训练集特征和目标值。
        X_val, y_val: 验证集特征和目标值。
        X_test, y_test: 测试集特征和目标值。
        scaler: 用于反归一化的 MinMaxScaler 对象。
    """
    # 从数据库获取数据
    data = db_handler.fetch_data()
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by="Date", inplace=True)

    # 取收盘价并归一化
    prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)

    # 分割数据为训练集和测试集
    test_cutoff = len(prices_scaled) - 252  # 最近一年约为252个交易日
    train_data = prices_scaled[:test_cutoff]
    test_data = prices_scaled[test_cutoff - window_size:]  # 包括窗口长度的数据

    # 定义滑动窗口生成函数
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    # 创建训练集和测试集序列
    X_train_full, y_train_full = create_sequences(train_data, window_size)
    X_test, y_test = create_sequences(test_data, window_size)

    # 从训练集中分割验证集
    val_size = int(len(X_train_full) * val_split)
    X_val, y_val = X_train_full[-val_size:], y_train_full[-val_size:]
    X_train, y_train = X_train_full[:-val_size], y_train_full[:-val_size]

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler
