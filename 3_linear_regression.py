import torch
from torch.nn import Linear, Module, MSELoss
# 网络训练所需模型
from torch.optim import SGD
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

x = np.linspace(0, 20, 500)
# 均匀生成0~20之间的数据500个
y = 5 * x + 7
plt.plot(x, y)
plt.show()
# 不加这句话不出图

x = np.random.rand(256)
# 0~1随机
noise = np.random.randn(256) / 4
# 0~1方差为0
y = 5 * x + 7 + noise
df = pd.DataFrame()
# 生成一个空的字典类型
df['x'] = x
# 字典x对应x的值
df['y'] = y
# 字典y对应y的值
sns.lmplot(x='x', y='y', data=df)
plt.show()
# 由于x是生成的随机数，不是从小到大的时间序列，所以只能用seaborn画散点图

model = Linear(1, 1)
# (1, 1)代表输入输出的特征(feature)数量都是1
criterion = MSELoss()
# 均方损失函数
optim = SGD(model.parameters(), lr=0.01)
# 优化器我们选择最常见的优化方法 SGD，就是每一次迭代计算 mini-batch 的梯度，然后对参数进行更新，学习率 0.01
epochs = 3000
# 训练3000次
x_train = x.reshape(-1, 1).astype('float32')
y_train = y.reshape(-1, 1).astype('float32')
# -1意为根据另一维度推测数组大小，训练数据为列向量

# 开始训练线性回归模型
for i in range(epochs):
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    # 训练时必须转换为tensor类型
    outputs = model(inputs)
    # 使用模型预测数据
    optim.zero_grad()
    # 梯度记得归零
    loss = criterion(outputs, labels)
    # 计算损失
    loss.backward()
    # 反向传播
    optim.step()
    # 优化器默认方法优化
    if i % 100 == 0:
        # 每100次打印一下损失函数
        print('epoch {}, loss {:1.4f}'.format(i, loss.data.item()))
[a, b] = model.parameters()
print(a.item(), b.item())
print(list(model.parameters()))

predicted = model(torch.from_numpy(x_train)).data.numpy()
# 预测的结果是一个支持autograd的张量，因此用detach()或者data去除求导条件，然后才能用numpy转化为数组
plt.plot(x_train, y_train, 'go', label='data', alpha=0.3)
plt.plot(x_train, predicted, label='predicted', alpha=1)
plt.legend()
plt.show()
