import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# 计算相对强弱指数 (RSI)
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# 计算MACD（指数平滑异同移动平均线）
def compute_macd(data, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = data['Close'].ewm(span=fast_period, min_periods=1, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_period, min_periods=1, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, min_periods=1, adjust=False).mean()

    return macd, signal


# 计算布林带
def compute_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)

    return upper_band, lower_band


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

    # 价格变化率
    data['Price_Change_Rate'] = data['Close'].pct_change()

    # 波动率
    data['Volatility'] = (data['High'] - data['Low']) / data['Close']

    # 成交量变化率
    data['Volume_Change_Rate'] = data['Volume'].pct_change()

    # 周期性特征
    data['Day_sin'] = np.sin(2 * np.pi * data['Date'].dt.dayofyear / 365)
    data['Day_cos'] = np.cos(2 * np.pi * data['Date'].dt.dayofyear / 365)

    # 计算RSI
    data['RSI'] = compute_rsi(data, window=14)

    # 计算MACD和Signal线
    data['MACD'], data['MACD_Signal'] = compute_macd(data)

    # 计算布林带
    data['Bollinger_Upper'], data['Bollinger_Lower'] = compute_bollinger_bands(data)

    # 填充新增特征中的缺失值
    data.fillna(method='bfill', inplace=True)

    # 选择多特征作为输入
    features = data[['Close', 'High', 'Low', 'Open', 'Volume', 'Price_Change_Rate', 'Volatility',
                     'Volume_Change_Rate', 'Day_sin', 'Day_cos', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Upper',
                     'Bollinger_Lower']].values

    # 单独对每个特征进行标准化
    scalers = {}
    features_scaled = np.zeros_like(features)
    feature_columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'Price_Change_Rate', 'Volatility',
                       'Volume_Change_Rate', 'Day_sin', 'Day_cos', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Upper',
                       'Bollinger_Lower']

    for i, column in enumerate(feature_columns):
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
