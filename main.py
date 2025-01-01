from src.database import MongoDBHandler
from src.preprocessing import load_and_preprocess_data
from src.n_beats import NBEATSModel
from src.train import train_model
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    db_handler = MongoDBHandler()
    window_size = 60
    forecast_horizon = 120
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess_data(db_handler, window_size)

    model = NBEATSModel(input_dim=window_size, forecast_horizon=forecast_horizon)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, learning_rate=0.0005, device=device)

    model.eval()
    predictions = []
    with torch.no_grad():
        current_input = X_test[0].unsqueeze(0)
        for _ in range(forecast_horizon):
            pred = model(current_input)
            predictions.append(pred[:, -1].cpu().numpy())
            current_input = torch.cat((current_input[:, 1:], pred[:, -1].unsqueeze(1)), dim=1)

    predictions = np.array(predictions).reshape(-1, 1)
    predictions_rescaled = scaler.inverse_transform(predictions)
    print(predictions_rescaled)
