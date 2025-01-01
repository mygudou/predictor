import numpy as np
import torch

def predict_future(model, scaler, initial_input, future_steps, noise_factor=0.01):
    predictions = []
    current_input = initial_input.copy()

    for _ in range(future_steps):
        with torch.no_grad():
            pred = model(torch.tensor(current_input, dtype=torch.float32)).item()
        predictions.append(pred)
        pred_array = np.array([[[pred + np.random.normal(0, noise_factor)]]])
        current_input = np.append(current_input[:, 1:, :], pred_array, axis=1)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
