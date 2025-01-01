from src.database import MongoDBHandler
from src.preprocessing import load_and_preprocess_data
from src.model import TimeSeriesTransformer
from src.train import train_model
from src.predict import predict_future
import torch

def main():
    db_handler = MongoDBHandler()

    window_size = 60
    X, y, scaler = load_and_preprocess_data(db_handler, window_size)

    model = TimeSeriesTransformer(input_dim=1, d_model=64, n_heads=4, num_layers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(model, X, y, device, epochs=50)

    future_steps = 126
    initial_input = X[-1].reshape(1, window_size, 1)
    predictions = predict_future(model, scaler, initial_input, future_steps, device)

    print(predictions)

if __name__ == "__main__":
    main()
