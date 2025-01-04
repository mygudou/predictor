import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd


def load_and_preprocess_data(db_handler, window_size):
    # 从 MongoDB 加载数据
    data = db_handler.fetch_data()

    # 确保数据按日期升序排序
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by="Date", inplace=True)

    # 计算新增特征
    # 移动平均线
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    # data['MA_60'] = data['Close'].rolling(window=60).mean()

    # 价格变化率
    data['Price_Change_Rate'] = data['Close'].pct_change()

    # 波动率
    data['Volatility'] = (data['High'] - data['Low']) / data['Close']

    # 成交量变化率
    data['Volume_Change_Rate'] = data['Volume'].pct_change()

    # 周期性特征
    data['Day_sin'] = np.sin(2 * np.pi * data['Date'].dt.dayofyear / 365)
    data['Day_cos'] = np.cos(2 * np.pi * data['Date'].dt.dayofyear / 365)

    # 填充新增特征中的缺失值
    data.fillna(method='bfill', inplace=True)

    # 选择多特征作为输入
    features = data[['Close', 'High', 'Low', 'Open', 'Volume', 'MA_5', 'MA_10', 'MA_20', 'Price_Change_Rate', 'Volatility',
                     'Volume_Change_Rate', 'Day_sin', 'Day_cos']].values

    # 单独对每个特征进行标准化
    scalers = {}
    features_scaled = np.zeros_like(features)
    for i, column in enumerate(
            ['Close', 'High', 'Low', 'Open', 'Volume', 'MA_5', 'MA_10', 'MA_20', 'Price_Change_Rate', 'Volatility',
             'Volume_Change_Rate', 'Day_sin', 'Day_cos']):
        scaler = StandardScaler()
        features_scaled[:, i] = scaler.fit_transform(features[:, i].reshape(-1, 1)).flatten()
        scalers[column] = scaler

    # 时间窗口划分
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size, 0])  # 预测收盘价
        return np.array(X), np.array(y)

    X, y = create_sequences(features_scaled, window_size)

    return X, y, scalers
