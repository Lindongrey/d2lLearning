import os
import torch
import pandas as pd

# 2.2.1.读取数据
# os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')  # 列名
#     f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')
data = pd.read_csv(data_file)
# print(data)

# 2.2.2.处理缺失值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs.iloc[:, 0] = inputs.iloc[:, 0].fillna(inputs.iloc[:, 0].mean())
inputs = pd.get_dummies(inputs, dummy_na=True)
# print(inputs)

# 2.2.3.转化为张量
X = torch.tensor(inputs.to_numpy(dtype=float))
print(X)
