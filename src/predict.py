import numpy as np
import torch


def predict_future(model, scaler, initial_input, future_steps, device, noise_factor=0.01):
    predictions = []
    current_input = initial_input.clone().to(device)

    with torch.no_grad():
        for _ in range(future_steps):
            pred = model(current_input)
            pred_scalar = pred[:, -1]  # Extract the last step prediction
            predictions.append(pred_scalar.cpu().numpy())
            noise = torch.tensor(np.random.normal(0, noise_factor), dtype=torch.float32).to(device)
            pred_array = (pred_scalar + noise).unsqueeze(0)
            current_input = torch.cat((current_input[:, 1:], pred_array.unsqueeze(1)), dim=1)

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)
