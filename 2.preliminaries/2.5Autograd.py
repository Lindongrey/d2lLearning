import torch

x = torch.tensor(2, dtype=torch.float32, requires_grad=True)
y = x * x + x * 2 + 1
y.backward()
print(x.grad)
