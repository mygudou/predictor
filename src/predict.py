import numpy as np
import torch


def predict_future(model, scaler, initial_input, future_steps, device="cpu"):
    model.to(device)
    model.eval()

    predictions = []
    current_input = torch.tensor(initial_input, dtype=torch.float32).to(device)

    for _ in range(future_steps):
        with torch.no_grad():
            pred = model(current_input).item()
        predictions.append(pred)

        pred_tensor = torch.tensor([[[pred]]], dtype=torch.float32).to(device)
        current_input = torch.cat([current_input[:, 1:, :], pred_tensor], dim=1)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
