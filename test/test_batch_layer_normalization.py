import torch
import torch.nn as nn

class SimpleNetBN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.linear2(x)

class SimpleNetLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.ln(x)
        x = self.relu(x)
        return self.linear2(x)

batch_size = 4
dim = 8
x = torch.randn(batch_size, dim)

model_bn = SimpleNetBN(dim)
model_ln = SimpleNetLN(dim)

out_bn = model_bn(x)
out_ln = model_ln(x)

print("BatchNorm output shape:", out_bn.shape)
print(out_bn)
print("LayerNorm output shape:", out_ln.shape)
print(out_ln)