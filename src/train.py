
import torch
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, device='cpu'):
    # 转换数据格式为 torch tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # 假设静态特征是一个常数向量（例如只有一个特征，长度与样本数相同）
    static_features_train = torch.tensor(np.ones((X_train.shape[0], 1)), dtype=torch.float32).to(device)
    static_features_val = torch.tensor(np.ones((X_val.shape[0], 1)), dtype=torch.float32).to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 训练
        output = model(X_train_tensor, static_features_train)  # 使用静态特征
        loss = F.mse_loss(output.squeeze(), y_train_tensor)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # 验证
    model.eval()
    with torch.no_grad():
        output_val = model(X_val_tensor, static_features_val)
        val_loss = F.mse_loss(output_val.squeeze(), y_val_tensor)
        print(f'Validation Loss: {val_loss.item():.4f}')
