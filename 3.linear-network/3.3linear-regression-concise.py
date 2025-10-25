import numpy as np
import torch
from d2l.torch import synthetic_data
from torch.utils import data
from d2l import torch as d2l
from torch import nn

# 构建数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 特征和目标值，也就是输入和理想输出
features, labels = synthetic_data(true_w, true_b, 1000)

# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
# print(next(iter(data_iter)))

# 定义模型
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
# normal_ 和 fill_ 重写了这两个参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(1)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
# 传入参数和学习速率
trainer = torch.optim.SGD(net.parameters(), lr=0.003)

# 训练
num_epoch = 30
for epoch in range(num_epoch):
    for X, y in data_iter:
        # 把结果和理想结果对比，得出损失函数
        l = loss(net(X), y)
        # 清空旧梯度
        trainer.zero_grad()
        # 计算新梯度
        l.backward()
        # 调整参数让梯度下降
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
