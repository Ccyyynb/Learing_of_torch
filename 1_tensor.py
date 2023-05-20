import torch
import numpy as np

var = torch.__version__
# check the version of torch
print(var)

print(torch.version.cuda)
# check the version of cuda

print(torch.cuda.get_device_name(0))
# get the name of GPU

var1 = torch.rand(3, 3)
# get a tensor
print(var1)

print(var1.shape)
# get the size of the tensor

xx = var1.view(1, 9)
print(xx.shape)
# reshape

var2 = torch.tensor(10)
# var2 is a scalar(标量)
print(var2.item())
# item can get the number of a scalar

var3 = torch.ones(3, 3)
# get a matrix full of 1
var4 = torch.zeros(3, 3)
# get a matrix full of 0
var5 = torch.eye(3, 3)
# get a unit matrix

max_value, max_id = torch.max(var1, dim=1)
print(max_value, max_id)
# get the max number and the place of the number,if dim=1 ->row    dim=0 ->list

sum_var5 = torch.sum(var5, dim=1)
print(sum_var5)
# summarize



