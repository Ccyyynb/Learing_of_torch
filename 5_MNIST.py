import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

BATCH_SIZE = 512
EPOCHS = 20
# 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 判断是否使用GPU

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=BATCH_SIZE, shuffle=True)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(1, 10, 5)  # 输入通道数1，输出通道数10，核的大小5
        self.conv2 = nn.Conv2d(10, 20, 3)  # 输入通道数10，输出通道数20，核的大小3
        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = nn.Linear(20 * 10 * 10, 500)  # 输入通道数是2000，输出通道数是500
        self.fc2 = nn.Linear(500, 10)  # 输入通道数是500，输出通道数是10，即10分类

    def forward(self, x):
        in_size = x.size(0)  # 在本例中in_size=512，也就是BATCH_SIZE的值。输入的x可以看成是512*1*28*28的张量。
        out = self.conv1(x)  # batch*1*28*28 -> batch*10*24*24（28x28的图像经过一次核为5x5的卷积，输出变为24x24）
        out = F.relu(out)  # batch*10*24*24（激活函数ReLU不改变形状））
        out = F.max_pool2d(out, 2, 2)  # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半）
        out = self.conv2(out)  # batch*10*12*12 -> batch*20*10*10（再卷积一次，核的大小是3）
        out = F.relu(out)  # batch*20*10*10
        out = out.view(in_size, -1)  # batch*20*10*10 -> batch*2000（out的第二维是-1，说明是自动推算，本例中第二维是20*10*10）
        out = self.fc1(out)  # batch*2000 -> batch*500
        out = F.relu(out)  # batch*500
        out = self.fc2(out)  # batch*500 -> batch*10
        out = F.log_softmax(out, dim=1)  # 计算log(softmax(x))
        return out

model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
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
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # 将一批的损失相加
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

for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)

