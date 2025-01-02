import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def load_and_preprocess_data(db_handler, window_size):
    # 从 MongoDB 加载数据
    data = db_handler.fetch_data()
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by="Date", inplace=True)

    # 选择收盘价作为目标数据
    prices = data['Close'].values.reshape(-1, 1)

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)

    # 时间窗口划分
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    X, y = create_sequences(prices_scaled, window_size)
    return X, y, scaler
