import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def load_and_preprocess_data(db_handler, window_size):
    # 从 MongoDB 加载数据
    data = db_handler.fetch_data()
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by="Date", inplace=True)

    # 特征选择（包括 `Close`, `High`, `Low`, `Open`, `Volume`）
    features = ['Close', 'High', 'Low', 'Open', 'Volume']
    data_features = data[features].values

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_features)

    # 时间窗口划分
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size, :])  # 添加所有特征
            y.append(data[i + window_size, 0])   # 收盘价为目标
        return np.array(X), np.array(y)

    X, y = create_sequences(data_scaled, window_size)
    return X, y, scaler
