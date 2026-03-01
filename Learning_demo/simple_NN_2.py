import torch
import torch.nn as nn
import torch.optim as optim  # 优化器（常和nn配合使用）

# 1. 定义模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.net(x)

# 2. 初始化组件
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)  # 模型迁移到GPU
criterion = nn.CrossEntropyLoss().to(device)  # 损失函数迁移到GPU
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器（关联模型参数）

# 3. 训练循环
model.train()  # 切换训练模式
for epoch in range(10):
    running_loss = 0.0
    # 模拟批量数据（实际用DataLoader）
    for i in range(100):
        # 生成模拟输入和标签
        inputs = torch.randn(32, 1, 28, 28).to(device)
        labels = torch.randint(0, 10, (32,)).to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播 + 优化器更新
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/100:.4f}")

# 4. 模型评估
model.eval()  # 切换评估模式
with torch.no_grad():  # 禁用梯度计算（节省内存）
    test_input = torch.randn(1, 1, 28, 28).to(device)
    pred = model(test_input)
    print(f"预测类别：{torch.argmax(pred).item()}")

# 5. 模型保存/加载
torch.save(model.state_dict(), "mlp_model.pth")  # 保存参数
mm = model.state_dict()
model.load_state_dict(torch.load("mlp_model.pth"))  # 加载参数
print(model)