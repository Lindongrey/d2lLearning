import torch
from torch import nn
from d2l import torch as d2l
import matplotlib
matplotlib.use('TkAgg')

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 三层的单元数
num_inputs, num_hiddens, num_outputs = 784, 256, 10

# 输入层不计算，单隐藏层和输出层才需要权重和偏置值
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]


# ReLU 激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(a, X)


# 定义模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)
    return (H@W2 + b2)


# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')


# 训练
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

# 预测
d2l.predict_ch3(net, test_iter)
d2l.plt.show()
