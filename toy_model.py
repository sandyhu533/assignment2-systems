import torch
from torch import nn
import torch.nn.functional as F

class ToyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False, device="cuda")
        self.ln = nn.LayerNorm(10, device="cuda")
        self.fc2 = nn.Linear(10, out_features, bias=False, device="cuda")
        self.relu = nn.ReLU()
    
    def forward(self, x):
        print(f'input {x.dtype}')
        x = self.fc1(x)
        print(f'after fc1 {x.dtype}')
        x = self.relu(x)
        print(f'after relu {x.dtype}')
        x = self.ln(x)
        print(f'after ln {x.dtype}')
        x = self.fc2(x)
        print(f'after fc2 {x.dtype}')
        return x

print(f'>>>>>>>>>>>before autocasting>>>>>>>>>>>>')

model = ToyModel(2,2)
print(f"param dtype: {next(model.parameters()).dtype}")

x = torch.zeros((2,), dtype=torch.float32, device = "cuda")
y = model(x)
print(f'logits {y.dtype}')
loss = F.cross_entropy(y, torch.ones((2,), device = "cuda"))
print(f'loss {loss.dtype}')
loss.backward()
print(f'fc1.grad: {model.fc1.weight.grad.dtype}')   # FP32
print(f'fc2.grad: {model.fc2.weight.grad.dtype}')   # FP32

print(f'>>>>>>>>>>>after autocasting>>>>>>>>>>>>')

with torch.autocast(device_type="cuda", dtype=torch.float16):
    model = ToyModel(2,2)
    print(f"param dtype: {next(model.parameters()).dtype}")

    x = torch.zeros((2,), dtype=torch.float32, device = "cuda")
    y = model(x)
    print(f'logits {y.dtype}')
    loss = F.cross_entropy(y, torch.ones((2,), device = "cuda"))
    print(f'loss {loss.dtype}')
    loss.backward()
    print(f'fc1.grad: {model.fc1.weight.grad.dtype}')   # FP32
    print(f'fc2.grad: {model.fc2.weight.grad.dtype}')   # FP32