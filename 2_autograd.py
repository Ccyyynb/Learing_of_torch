import torch
print(torch.__version__)

x = torch.rand(5, 5, requires_grad=True)
y = torch.rand(5, 5, requires_grad=True)
# 默认为不会自动求导，若要开启自动求导需在生成tensor时加入语句
z = torch.sum(x + y)
# pytorch会自动记录关于tensor的操作，当计算完成后调用.backward()方法自动计算梯度并将结果保存在grad属性中
z.backward()
# 如果Tensor类表示的是一个标量（即它是包含一个元素的张量），则不需要为backward()指定任何参数，但是如果它有更多的元素，则需要指定一个gradient参数，它是形状匹配的张量。
print(x.grad, y.grad)

# 需要指定参数的自动求导示例
x = torch.rand(5, 5, requires_grad=True)
y = torch.rand(5, 5, requires_grad=True)
z = x**2+y**3
z.backward(torch.ones_like(x))
print(x.grad)
# 我们的返回值不是一个标量，所以需要输入一个大小相同的张量作为参数，这里我们用ones_like函数根据x生成一个张量
