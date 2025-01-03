import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, X_train, y_train, X_val, y_val, static_features_train, static_features_val, epochs, batch_size=32, device='cpu'):
    # 转换数据格式为 torch tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # 假设静态特征是一个常数向量（例如只有一个特征，长度与样本数相同）
    static_features_train = torch.tensor(static_features_train, dtype=torch.float32).to(device)
    static_features_val = torch.tensor(static_features_val, dtype=torch.float32).to(device)

    # 创建 DataLoader
    train_data = TensorDataset(X_train_tensor, y_train_tensor, static_features_train)
    val_data = TensorDataset(X_val_tensor, y_val_tensor, static_features_val)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 训练
        for X_batch, y_batch, static_batch in train_loader:
            # Forward pass
            output = model(X_batch, static_batch)  # 使用静态特征
            loss = F.mse_loss(output.squeeze(), y_batch)

            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # 验证
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for X_batch, y_batch, static_batch in val_loader:
            output_val = model(X_batch, static_batch)
            val_loss += F.mse_loss(output_val.squeeze(), y_batch).item()

        print(f'Validation Loss: {val_loss / len(val_loader):.4f}')
