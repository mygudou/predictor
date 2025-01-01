import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, X_train, y_train, epochs=100, batch_size=64, learning_rate=0.0005, device="cuda"):
    # 将数据转换为 PyTorch 张量并移动到指定设备
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

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

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
