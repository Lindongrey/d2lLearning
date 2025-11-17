import torch
from torch import nn
from d2l import torch as d2l


# 多输入通道的卷积运算
def corr2d_multi_in(X, K):
    return sum(d2l.corr2d(X, K) for X, K in zip(X, K))


# 多通道输入输出的卷积运算
def corr2d_multi_in_ous(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


# 1 * 1 卷积层
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X.reshape((c_i, h * w))
    K.reshape((c_o, c_i))
    Y = torch.matmul(X, K)
    return Y.reshape((c_o, h, w))
