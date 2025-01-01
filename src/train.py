import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=64, learning_rate=0.0005, device="cuda"):
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    criterion = nn.SmoothL1Loss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = criterion(val_output.squeeze(), y_val).item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
