import torch

# 2.1.2.运算符
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
Z = torch.cat((X, Y), dim=0)

# print(Z)
# print(X == Y)
# print(X.sum())

# 2.1.3.广播机制
X = torch.arange(3)
Y = torch.arange(3).reshape(-1, 1)
# print(X + Y)

# 2.1.4.索引和切片
print(X[1:3])
print(X[:])
