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

BATCH_SIZE = 64
EPOCHS = 14
learnrate = 0.0003
# 总共训练批次
torch.manual_seed(3407)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 判断是否使用GPU

train_transform = transforms.Compose([
    transforms.Resize(112),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(degrees=(-10, 10)),
    # transforms.RandomResizedCrop(size=(32, 32)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.Normalize(0.485, 0.229)
    # transforms.RandomErasing(p=0.5, scale=(0.2, 0.33))
    # 随机擦除

])
test_transform = transforms.Compose([
    transforms.Resize(112),
    # transforms.RandomHorizontalFlip(0.5),
    # transforms.Resize(48),
    # transforms.CenterCrop(32),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transforms.Normalize(0.485, 0.229)
])
val_transform = transforms.Compose([
    transforms.Resize(112),
    # transforms.RandomHorizontalFlip(),
    # transforms.Resize(48),
    # transforms.CenterCrop(32),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transforms.Normalize(0.485, 0.229)
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

        # 将这个名字和路径进行拼接 就能得到这个图片的相对路径
        # img = Image.open(img_item_path).convert('RGB')
        img = Image.open(img_item_path).convert('L')
        # 得到这个img
        # fp = open(img_item_path, 'rb')
        # img = Image.open(fp)
        # fp.close()
        '''
        img = cv.imread(img_item_path, 1)
        # cv读图路径不能有中文！！！！！！！
        img = np.asarray(img)
        # 必须转换为numpy数组或者PIL才能transform
        '''
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


class Vgg16_net(nn.Module):

    def __init__(self):
        super(Vgg16_net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1=32   32*32*64
            nn.BatchNorm2d(64),
            # inplace-选择是否进行覆盖运算
            # 意思是是否将计算得到的值覆盖之前的值，比如
            nn.ReLU(inplace=True),
            # 意思就是对从上层网络Conv2d中传递下来的tensor直接进行修改，
            # 这样能够节省运算内存，不用多存储其他变量

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # (32-3+2)/1+1=32    32*32*64
            # Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，
            # 一方面使得数据分布一致，另一方面避免梯度消失。
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2,stride=2)   # (32-2)/2+1=16         16*16*64
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # (16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # (16-3+2)/1+1=16   16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)    # (16-2)/2+1=8     8*8*128
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),


            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)     # (8-2)/2+1=4      4*4*256
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)    # (4-2)/2+1=2     2*2*512
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2     2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2      2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)   # (2-2)/2+1=1      1*1*512
        )

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc = nn.Sequential(
            # y=xA^T+b  x是输入,A是权值,b是偏执,y是输出
            # nn.Liner(in_features,out_features,bias)
            # in_features:输入x的列数  输入数据:[batchsize,in_features]
            # out_freatures:线性变换后输出的y的列数,输出数据的大小是:[batchsize,out_features]
            # bias: bool  默认为True
            # 线性变换不改变输入矩阵x的行数,仅改变列数
            nn.Linear(512*3*3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.conv(x)
        # 这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成512列
        # 那不确定的地方就可以写成-1

        # 如果出现x.size(0)表示的是batchsize的值
        # x=x.view(x.size(0),-1)
        # print(x.size())
        x = x.view(-1, 512*3*3)
        x = self.fc(x)
        return x

model = Vgg16_net().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=learnrate, betas=(0.9, 0.999), eps=1e-08)
# optimizer = optim.SGD(model.parameters(), lr=0.003)
losses = []
train_losses = []
test_losses = []
val_losses = []
test_acc = []
train_acc = []
val_acc = []
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct_t = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target.long())
        train_loss += F.cross_entropy(output, target.long(), reduction='sum').item()
        # 不加long计算交叉熵时会报错
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().data.item())
        pred = output.max(1, keepdim=True)[1]
        correct_t += pred.eq(target.view_as(pred)).sum().item()
        correct = pred.eq(target.view_as(pred)).sum().item()
        if(batch_idx+1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}，Accuracy: ({:.4f}%)'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), 100. * correct / len(data)))
    train_acc.append(correct_t / len(train_loader.dataset))
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target.long(), reduction='sum').item()
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
    test_losses.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc.append(correct / len(test_loader.dataset))

def val(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.cross_entropy(output, target.long(), reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    print('\nval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    val_acc.append(correct / len(val_loader.dataset))



for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)
    val(model, DEVICE, val_loader)


plt.xkcd()
plt.plot(losses)
plt.xlabel('Batch')
plt.ylabel('loss')
plt.show()

plt.xkcd()
plt.plot(train_losses)
plt.plot(test_losses)
plt.plot(val_losses)
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'test_loss', 'val_loss'])
plt.show()

plt.xkcd()
plt.plot(test_acc)
plt.plot(train_acc)
plt.plot(val_acc)
plt.xlabel('Epoch')
plt.ylabel('acc')
plt.legend(['test_acc', 'train_acc', 'val_acc'])
plt.show()
print('\n', max(test_acc))
print('\n', max(train_acc))
print('\n', max(val_acc))
torch.save(model.state_dict(), "VGG16params.pkl")