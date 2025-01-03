import torch
import torch.optim as optim
import torch.nn as nn


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=100, batch_size=32, learning_rate=0.0005,  # 降低学习率
                weight_decay=1e-5, patience=12, device='cpu'):  # 增加 patience
    # 转换数据为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # 优化器与损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # 设置学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    best_val_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output.squeeze(), y_val_tensor).item()

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")

        # 学习率调度
        scheduler.step(val_loss)

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), "best_time_series_model.pth")
            print("Model improved. Saving best model.")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs.")

        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
