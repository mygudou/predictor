import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(db_handler, window_size):
    # 从 MongoDB 加载数据
    data = db_handler.fetch_data()

    # 确保数据按日期升序排序
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by="Date", inplace=True)

    # 计算新增特征
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_8'] = data['Close'].rolling(window=8).mean()
    data['MA_21'] = data['Close'].rolling(window=21).mean()
    data['MA_55'] = data['Close'].rolling(window=55).mean()
    data['MA_144'] = data['Close'].rolling(window=144).mean()
    data['MA_233'] = data['Close'].rolling(window=233).mean()
    data['Price_Change_Rate'] = data['Close'].pct_change()
    data['Volatility'] = (data['High'] - data['Low']) / data['Close']
    data['Volume_Change_Rate'] = data['Volume'].pct_change()
    data['Day_sin'] = np.sin(2 * np.pi * data['Date'].dt.dayofyear / 365)
    data['Day_cos'] = np.cos(2 * np.pi * data['Date'].dt.dayofyear / 365)
    data.fillna(method='bfill', inplace=True)

    # 动态特征选择
    dynamic_features = ['Close', 'High', 'Low', 'Open', 'Volume', 'MA_5', 'MA_8', 'MA_21', 'MA_55', 'MA_144', 'MA_233',
                        'Price_Change_Rate', 'Volatility', 'Volume_Change_Rate', 'Day_sin', 'Day_cos']

    features = data[dynamic_features].values

    # 对每个特征进行标准化
    scalers = {}
    features_scaled = np.zeros_like(features)
    for i, column in enumerate(dynamic_features):
        scaler = StandardScaler()
        features_scaled[:, i] = scaler.fit_transform(features[:, i].reshape(-1, 1)).flatten()
        scalers[column] = scaler

    # 静态特征 (假设只有一个静态特征，这里可以根据实际情况扩展)
    static_features = np.array([1] * len(features)).reshape(-1, 1)  # 静态特征，假设只有一个特征，长度与数据集相同
    static_scaler = StandardScaler()
    static_features_scaled = static_scaler.fit_transform(static_features)

    # 时间窗口划分
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size, 0])  # 预测收盘价
        return np.array(X), np.array(y)

    X, y = create_sequences(features_scaled, window_size)

    return X, y, scalers, static_features_scaled
