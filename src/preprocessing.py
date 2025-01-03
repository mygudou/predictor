import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_and_preprocess_data(db_handler, window_size):
    # 从 MongoDB 加载数据
    data = db_handler.fetch_data()
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by="Date", inplace=True)

    # 选择多特征作为输入
    features = data[['Close', 'High', 'Low', 'Open']].values

    # 使用 StandardScaler 进行标准化
    scalers = [StandardScaler() for _ in range(features.shape[1])]
    features_scaled = np.zeros_like(features)
    for i in range(features.shape[1]):
        features_scaled[:, i] = scalers[i].fit_transform(features[:, i].reshape(-1, 1)).flatten()

    # 时间窗口划分
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])  # X is a list of sequences
            y.append(data[i + window_size, 0])  # y is the predicted target (Close price)
        X = np.array(X)  # Convert list of sequences to a numpy array
        y = np.array(y)  # Convert target values to a numpy array
        return X, y

    X, y = create_sequences(features_scaled, window_size)
    return X, y, scalers

