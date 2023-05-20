import os
from os.path import join
from typing import List
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2 as cv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 125
EPOCHS = 10
# 总共训练批次
torch.manual_seed(1234)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 判断是否使用GPU

train_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomResizedCrop(48),
    transforms.ToTensor(),
    # transforms.Normalize()
])
test_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.Resize(48),
    # transforms.CenterCrop(32),
    transforms.ToTensor(),
    # transforms.Normalize()
])
val_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.Resize(48),
    # transforms.CenterCrop(32),
    transforms.ToTensor(),
    # transforms.Normalize()
])

target_path = r'C:\Users\HP\Desktop\graduate_student\code\fer2013\datasets'
all_content = os.listdir(target_path)
print('All content numbers is', len(all_content))
n1 = 0
for content in all_content:
    if os.path.isdir(join(target_path, content)):
        content2 = os.listdir(join(target_path, content))
        for i in content2:
            content3 = os.listdir(join(join(target_path, content), i))
            n1 += len(content3)
    print('{} has {}'.format(content, n1))
    n1 = 0
# 统计三层文件夹数据集各部分数据数量


class fer2013(Dataset):
    def __init__(self, root_dir, label_dir, trans, transform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        self.transform = trans
        # 将这个路径下的文件变成一个列表的形式

    def __getitem__(self, idx):
        # 想要获取每一个图片
        img_name = self.img_path[idx]
        # 在这个列表下 用idx看是第几个图
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        '''
        # 将这个名字和路径进行拼接 就能得到这个图片的相对路径
        # img = Image.open(img_item_path)
        # 得到这个img
        # fp = open(img_item_path, 'rb')
        # img = Image.open(fp)
        # fp.close()
        '''
        img = cv.imread(img_item_path, cv.IMREAD_GRAYSCALE)
        # cv读图路径不能有中文！！！！！！！
        img = np.asarray(img)
        # 必须转换为numpy数组或者PIL才能transform
        label = self.label_dir
        label = np.asarray(eval(label))
        # label不能为元组，会没有‘to’属性，不能扔给GPU，改为np格式
        return self.transform(img), torch.tensor(label)

    def __len__(self):
        return len(self.img_path)

root_dir = r'C:\\Users\\HP\\Desktop\\graduate_student\\code\\fer2013\\datasets\\train'
train_dataset = []
for label_dir in ['0', '1', '2', '3', '4', '5', '6']:
    train_dataset += fer2013(root_dir, label_dir, train_transform)
# img, label = train_dataset[1]
# print(label, img)
# 加载训练集
root_dir = r'C:\\Users\\HP\\Desktop\\graduate_student\\code\\fer2013\\datasets\\test'
test_dataset = []
for label_dir in ['0', '1', '2', '3', '4', '5', '6']:
    test_dataset += fer2013(root_dir, label_dir, test_transform)
# 加载测试集
root_dir = r'C:\\Users\\HP\\Desktop\\graduate_student\\code\\fer2013\\datasets\\val'
val_dataset = []
for label_dir in ['0', '1', '2', '3', '4', '5', '6']:
    val_dataset += fer2013(root_dir, label_dir, val_transform)
# 加载验证集

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
# loader不支持下标索引

'''
print(train_loader)
for data in train_loader:
    pic, targets = data
    print(pic)
# 查看dataloader的数据
'''
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # batch*1*48*48（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是48x48）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(1, 10, 5)  # 输入通道数1，输出通道数10，核的大小5
        self.conv2 = nn.Conv2d(10, 20, 3)  # 输入通道数10，输出通道数20，核的大小3
        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = nn.Linear(20 * 20 * 20, 500)  # 输入通道数是2000，输出通道数是500
        self.fc2 = nn.Linear(500, 7)  # 输入通道数是500，输出通道数是10，即10分类

    def forward(self, x):
        in_size = x.size(0)  # 在本例中in_size=512，也就是BATCH_SIZE的值。输入的x可以看成是512*1*48*48的张量。
        out = self.conv1(x)  # batch*1*48*48 -> batch*10*44*44（28x28的图像经过一次核为5x5的卷积，输出变为44x44）
        out = F.relu(out)  # batch*10*24*24（激活函数ReLU不改变形状））
        out = F.max_pool2d(out, 2, 2)  # batch*10*44*44 -> batch*10*22*22（2*2的池化层会减半）
        out = self.conv2(out)  # batch*10*22*22 -> batch*20*20*20（再卷积一次，核的大小是3）
        out = F.relu(out)  # batch*20*20*20
        out = out.view(in_size, -1)  # batch*20*20*20 -> batch*8000（out的第二维是-1，说明是自动推算，本例中第二维是20*20*20）
        out = self.fc1(out)  # batch*8000 -> batch*500
        out = F.relu(out)  # batch*500
        out = self.fc2(out)  # batch*500 -> batch*7
        out = F.log_softmax(out, dim=1)  # 计算log(softmax(x))
        return out

model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08)
losses = []
test_acc = []
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target.long())
        # 不加long计算交叉熵时会报错
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().data.item())
        if(batch_idx+1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target.long(), reduction='sum').item()
            # 将一批的损失相加 不加long计算交叉熵时会报错
            pred = output.max(1, keepdim=True)[1]
            # 找到概率最大的下标
            # torch.max(0)和 torch.max(1)分别是找出tensor里每列/每行中最大的值，并返回索引（即为对应的预测数字）
            # 此处就是给出一张图片对应每种类别的概率的最大值的索引，即为该张图片所属的类别。
            # 而前面的“_,”的意思则是，torch.max()这个函数返回的是两个值，第一个值是具体的value（我们用下划线_表示）
            # 第二个值是value所在的index（也就是pred），而我们只关心第二个值，也就是pred，因为后面要用pred值来与label进行比较，
            # 来表示预测的正确与否，第一个值（概率）对我们而言并不重要，所以用下划线代替（当然可以用别的东西代替），习惯上都是用下划线来表示不关心的类别。
            correct += pred.eq(target.view_as(pred)).sum().item()
            # torch.eq()是比较input和output的函数,input必须为tensor类型，output可以为相同大小的tensor也可以为某个值
            # 当input和output都为tensor类型时，比较对应位置数字是否相同，相同则为1，否则为0。而torch.eq()函数得到的是个tensor。

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc.append(correct / len(test_loader.dataset))

for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)

plt.xkcd()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(losses)
plt.show()
plt.xkcd()
plt.xlabel('Epoch')
plt.ylabel('test_acc')
plt.plot(test_acc)
plt.show()

