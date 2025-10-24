import random
import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l

# 3.2.1.生成数据集
# 由于是学习，所以这个数据集的权重和偏移值已经确定了，分别是 [2, -3.4] 和 4.2
# 我们之后需要做的是，看看训练出来的权重和偏移值跟预设的值是否接近
# 方法就是线性回归
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 特征和目标值，也就是输入和理想输出
features, labels = synthetic_data(true_w, true_b, 1000)

# print('features:', features[0],'\nlabel:', labels[0])
# d2l.set_figsize()
# d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);
# plt.show()

# 3.2.2.读取数据集
def data_iter(batch_size, features, labels):
    # 样本总数
    num_examples = len(features)
    # 所有样本索引
    indices = list(range(num_examples))
    # 打乱样本索引
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        # min 函数防止最后一批不足 batch_size
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        # 返回一个二元组，包括随机切片后的特征和目标
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
    # print(X, " ", y)
    # break

# 3.2.3.初始化模型参数
# 注意我们的数据集是用两个预设好的参数生成的，但是训练的时候我们得用自己的参数，先随机生成
# 权重从均值 0.1，标准差 0.01 的正态分布中采样
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
# 偏移值初始化为 0
b = torch.zeros(1, requires_grad=True)

# 3.2.4.定义模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b

# 3.2.5.定义损失函数
# 使用平方损失函数，它的前提是预测值需要和真实值有一样的形状
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 3.2.6.定义优化算法
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 3.2.7.训练
# 学习速率
lr = 0.003
# 迭代次数
num_epochs = 30
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
