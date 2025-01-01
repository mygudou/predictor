import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=64, learning_rate=0.0005, device="cuda"):
    """
    训练模型，并支持验证集评估。

    参数:
        model: 待训练的 PyTorch 模型。
        X_train, y_train: 训练集特征和目标值。
        X_val, y_val: 验证集特征和目标值（可选）。
        epochs: 训练轮次，默认 100。
        batch_size: 批量大小，默认 64。
        learning_rate: 学习率，默认 0.0005。
        device: 设备 ("cuda" 或 "cpu")，默认 "cuda"。
    """
    # 将数据转换为 PyTorch 张量并移动到指定设备
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    if X_val is not None and y_val is not None:
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # 创建 DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    # 初始化优化器、学习率调度器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    criterion = nn.SmoothL1Loss()

    # 模型训练
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            # 前向传播
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            # 记录损失
            epoch_loss += loss.item()

        # 更新学习率
        scheduler.step()

        # 验证集评估
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_tensor)
                val_loss = criterion(val_output.squeeze(), y_val_tensor).item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
