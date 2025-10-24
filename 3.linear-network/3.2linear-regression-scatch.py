import random
import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l

# 生成数据集
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

# 读取数据集
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

for X, y in data_iter(batch_size, features, labels):
    print(X, " ", y)
    break
