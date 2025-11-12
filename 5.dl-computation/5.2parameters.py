import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
X = torch.rand(size=(2, 4))

# 参数访问
# print(net[2].state_dict())  # 第二个全连接层
# print(net[0].state_dict())  # 第一个全连接层，index 为 1 时是激活函数的层
# print(type(net[2].bias))
# print(net[2].bias)

# print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# print(*[(name, param.shape) for name, param in net.named_parameters()])
# print(net.state_dict()['2.bias'].data)  # 第二层的偏置


def block1():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU()
    )


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
# print(rgnet)
# print(rgnet[0][1][0].bias.data)  # 第一个块的第二个子块的第一个偏置


# 参数初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


# xavier 初始化
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


# 用常数 42 初始化
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


net[0].apply(init_xavier)
net[2].apply(init_42)
# print(net[0].weight.data[0])
# print(net[2].weight.data)


# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init ", *[(name, param.shape) for name, param in net.parameters()][0])
    nn.init.uniform_(m.weight, -10, 10)
    m.weight.data *= m.weight.data.abs() >= 5


# 参数绑定
shared = nn.Linear(8, 8)
net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    shared,
    nn.ReLU(),
    nn.Linear(8, 1)
)
