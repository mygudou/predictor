import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, X_train, y_train, epochs=50, batch_size=32, learning_rate=0.001, device="cuda"):
    """
    训练时间序列预测模型。

    :param model: PyTorch 模型实例
    :param X_train: 输入数据 (numpy array)
    :param y_train: 标签数据 (numpy array)
    :param epochs: 训练轮数
    :param batch_size: 批次大小
    :param learning_rate: 学习率
    :param device: 使用的设备 ("cuda" 或 "cpu")
    """
    # 转换数据为 PyTorch 张量并加载到指定设备
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    # 优化器与损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 数据加载器
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    # 将模型加载到指定设备
    model = model.to(device)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            # 前向传播
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            # 反向传播
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    # 保存训练好的模型
    torch.save(model.state_dict(), "time_series_model.pth")
