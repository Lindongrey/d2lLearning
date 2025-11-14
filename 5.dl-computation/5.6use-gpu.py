import torch
from torch import nn

# 查询支持和设备
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())


def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


x = torch.tensor([1, 2, 3], device=try_gpu(0), dtype=torch.float32)
# print(x.device)

# 复制存在 gpu0 上的张量
Z = x.cuda(0)
# print(Z)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
net(x)
print(net[0].weight.data.device)
