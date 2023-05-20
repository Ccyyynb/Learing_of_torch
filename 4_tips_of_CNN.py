import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import torchvision.utils
import torch.utils.data as data
# 对数据进行batch的划分，在训练模型时用到此函数，用来把训练数据分成多个小组，此函数每次抛出一组数据，直至把所有数据都抛出
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F


# 通过LeNet的结构来熟悉构建网络的函数及过程
class Reshape(torch.nn.Module):
    def forward(self, x):
        # 将图像转为(X,1,28,28)的形式，其中X为图像的数量，1*28*28为图像格式，1为通道数
        return x.view(-1, 1, 28, 28)


net = nn.Sequential(
    # 定义LeNet-5网络结构
    Reshape(), nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)


# Sequential用来进行模型设计
# nn.Sequential是一个有序的容器，神经网络模块将按照传入构造起的顺序依次被添加到计算图中执行
# nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)二维卷积函数
# in_channels 输入图像通道数 out_channels 卷积产生的通道数 kernel_size 卷积核尺寸，如果只输入一个int默认为(int,int)
# stride 卷积步长 padding 填充的数目(单位是圈，一圈两圈) padding_mode默认为零填充 bias 在输出中添加一个可学习的偏差
# nn.AvgPool2d() 二维池化函数 kernel_size 池化核尺寸 stride 步长
# nn.Sigmoid() 添加激活函数
# nn.Flatten() 把数据拉成一维数据

def load_data_fashion_mnist(batch_seze):
    # 下载MNIST数据集然后加载到内存中
    train_dataset = datasets.MNIST(root='../data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='../data', train=False, transform=transforms.ToTensor(), download=True)
    return (data.DataLoader(train_dataset, batch_size, shuffle=True),
            data.DataLoader(test_dataset, batch_size, shuffle=False))


# datasets.数据集名字  参数为 (保存路径，下载的是训练集还是验证集，数据格式，若数据集不存在是否允许下载数据集)
# transform=transforms.Totensor()代表将原始的PILImage格式或者numpy.array格式的数据格式化为可被pytorch快速处理的张量类型。
# data.DataLoader的参数为（数据源，每批数据的数量，是否打乱顺序）

batch_size = 64
train_iter, test_iter = load_data_fashion_mnist(batch_size)


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 损失函数
loss_function = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(net.parameters())
# 开始训练
num_epochs = 10
train_loss = []
for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(train_iter):
        out = net(x)
        y_onehot = F.one_hot(y, num_classes=10).float()  # 转为one-hot编码
        loss = loss_function(out, y_onehot)  # 均方差
        # 清零梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

# 绘制损失曲线
plt.figure(figsize=(16, 8))
plt.grid(True, linestyle='--', alpha=0.5)
plt.plot(train_loss, label='loss')
plt.legend(loc="best")
plt.show()

# 测试准确率
total_correct = 0
for batch_idx, (x, y) in enumerate(test_iter):
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_iter.dataset)
test_acc = total_correct / total_num
print(total_correct, total_num)
print("test acc:", test_acc)
