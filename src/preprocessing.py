import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def load_and_preprocess_data(db_handler, window_size):
    data = db_handler.fetch_data()
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by="Date", inplace=True)

    prices = data['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)

    # 分割训练集和测试集
    train_cutoff = len(prices_scaled) - 252  # 最近一年约为252个交易日
    train_data = prices_scaled[:train_cutoff]
    test_data = prices_scaled[train_cutoff - window_size:]  # 包括一个窗口长度的训练数据

    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data, window_size)
    X_test, y_test = create_sequences(test_data, window_size)

    return X_train, y_train, X_test, y_test, scaler
