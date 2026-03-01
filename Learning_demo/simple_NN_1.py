import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc0 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = Net()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 生成示例数据
inputs = torch.randn(100, 10)
targets = torch.rand(100, 1)

# 训练模型
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 评估模型
with torch.no_grad():
    test_inputs = torch.randn(10, 10)
    predictions = model(test_inputs)
    print("Test Predictions:", predictions)

print(model)
print(model.state_dict())
mm = model.state_dict()

par = model.parameters()
for p in par:
    print(p)

