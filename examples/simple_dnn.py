import torch
import torch.nn as nn
from flops_profiler import FLOPsProfiler


class SimpleDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


model     = SimpleDNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ✅ 追加するだけ
profiler = FLOPsProfiler(model)

for epoch in range(3):
    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))

    optimizer.zero_grad()

    # ✅ withで囲むだけ
    with profiler.profile():
        loss = criterion(model(x), y)
        loss.backward()

    optimizer.step()

    print(f"\n--- Epoch {epoch+1} ---")
    profiler.summary()
    profiler.reset()  # 次のエポックのためにリセット