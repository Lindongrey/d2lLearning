import torch

# 2.3.3.矩阵
X = torch.arange(20, dtype=torch.float32).reshape(4, 5)
print(X)
# print(X.T)

# 2.3.6.降维
# print(X.sum())
# print(X.sum(axis=0))
# print(X.sum(axis=[0, 1]))
# print(X.mean())
# print(X.mean(axis=1))

# 2.3.6.1.非降维求和
# print(X.sum(axis=1, keepdim=True))
# print(X / X.sum(axis=1, keepdim=True))
# print(X.cumsum(axis=0))

# 2.3.8.矩阵-向量积
x = torch.arange(4, dtype=torch.float32)
# print(torch.mv(X.T, x))

# 2.3.9.矩阵-矩阵乘法
Y = torch.ones(5, 4)
# print(torch.mm(X, Y))

# 2.3.10.范数
# L2 范数
print(torch.norm(X))
# L1 范数
print(torch.abs(X).sum())
