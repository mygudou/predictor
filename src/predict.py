import numpy as np
import torch

def predict_future(model, scalers, initial_input, future_steps, device='cpu'):
    model.eval()
    predictions = []
    current_input = initial_input.copy()

    for _ in range(future_steps):
        with torch.no_grad():
            current_tensor = torch.tensor(current_input, dtype=torch.float32).to(device)
            preds = model(current_tensor).cpu().numpy().flatten()
        predictions.extend(preds)

        # 滑动窗口更新
        for pred in preds:
            new_input = current_input[:, -1, :].copy()
            new_input[0, 0] = pred
            pred_array = new_input.reshape(1, 1, -1)
            current_input = np.append(current_input[:, 1:, :], pred_array, axis=1)

    predictions = np.array(predictions).reshape(-1, 1)
    extended_predictions = np.zeros((len(predictions), initial_input.shape[2]))
    extended_predictions[:, 0] = predictions[:, 0]

    for i in range(1, extended_predictions.shape[1]):
        extended_predictions[:, i] = initial_input[0, -1, i]

    for i, scaler in enumerate(scalers):
        extended_predictions[:, i] = scaler.inverse_transform(
            extended_predictions[:, i].reshape(-1, 1)
        ).flatten()

    return extended_predictions[:, 0]
