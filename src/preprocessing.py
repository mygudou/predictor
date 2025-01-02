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

    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    X, y = create_sequences(prices_scaled, window_size)
    return X, y, scaler