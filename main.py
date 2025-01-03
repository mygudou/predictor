from src.database import MongoDBHandler
from src.preprocessing import load_and_preprocess_data
from src.model import TimeSeriesTransformer
from src.train import train_model
from src.predict import predict_future
import torch

def main():
    db_handler = MongoDBHandler()

    window_size = 60
    X, y, scalers = load_and_preprocess_data(db_handler, window_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TimeSeriesTransformer(input_dim=4, d_model=128, n_heads=4, num_layers=2, future_steps=12).to(device)

    train_model(model, X, y, epochs=50, device=device)

    future_steps = 12
    initial_input = X[-1].reshape(1, window_size, 4)
    predictions = predict_future(model, scalers, initial_input, future_steps, device=device)

    print(predictions)


if __name__ == "__main__":
    main()
