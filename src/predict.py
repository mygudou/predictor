import numpy as np
import torch

def predict_future(model, scaler, initial_input, future_steps, device):
    predictions = []
    current_input = initial_input.copy()

    model.to(device)
    model.eval()

    for _ in range(future_steps):
        with torch.no_grad():
            current_tensor = torch.tensor(current_input, dtype=torch.float32).to(device)
            pred = model(current_tensor).item()
        predictions.append(pred)

        pred_array = np.array([[[pred]]])
        current_input = np.append(current_input[:, 1:, :], pred_array, axis=1)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
